import os
import json
from argparse import ArgumentParser
from sysconfig import get_path
from typing import List
import copy

from sympy import threaded
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model, safe_open, load_file
import time
from deepseek.model import Transformer, ModelArgs
from vptq.layers.vqlinear import VQuantLinear
from deepseek.model import ColumnParallelLinear, ColumnParallelVQLinear, RowParallelLinear, RowParallelVQLinear, Linear
from torch import nn
from tqdm import tqdm

def find_layers(module, target_layers=[nn.Linear], name=''):
    if type(module) in target_layers:
        return {name: module}
    res = {}
    for old_name, child in module.named_children():
        res.update(find_layers(child, target_layers=target_layers, name=name + '.' + old_name if name != '' else old_name))
    return res

def replace_layer(module, target_name, layer, module_name=None):
    for child_name, child_module in module.named_children():
        current_name = child_name if module_name is None else f'{module_name}.{child_name}'
        if target_name == current_name:
            setattr(module, child_name, layer)
            return True 
        else:
            if replace_layer(child_module, target_name, layer, current_name):
                return True 
    return False

def convert_str_to_dtypes(obj):
    """Recursively convert string representations of torch.dtype objects back to torch.dtype in nested dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: convert_str_to_dtypes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_str_to_dtypes(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('torch.'):
        try:
            # Extract the dtype part from the string, create the dtype object using getattr
            return getattr(torch, obj.split('.')[-1])
        except AttributeError:
            # If the dtype string is not a valid torch dtype, return the string as is
            return obj
    else:
        return obj

# load layer config
def get_layer_config(quant_config: str) -> list[tuple[int, str, str]]:
    with open(quant_config, 'r') as f:
        quant_config = json.load(f)
    quant_config = quant_config['quantization_config']['config_for_layers']
    return quant_config

def get_quantized_deepseek(model, ckpt_path, quant_config, 
                           world_size: int=1, rank: int=0,
                           dry_run: bool=False, dtype=torch.bfloat16):
    num_layers = len(model.layers)
    layers = model.layers
    quant_config = get_layer_config(quant_config)

    target_layers = [ColumnParallelLinear, RowParallelLinear, Linear]
    
    for layer_idx in tqdm(range(num_layers), desc="Initializing"):
        # hack, all layers are the same vector length and num centroids
        if layer_idx <= 3:
            ops = find_layers(layers[layer_idx], target_layers)
            for op_name, op in ops.items():
                op_name = f'layers.{layer_idx}.{op_name}'
                op_args = quant_config[op_name]
                op_args = convert_str_to_dtypes(op_args)
                if isinstance(op, Linear):
                    vqlinear = VQuantLinear(**op_args)
                elif isinstance(op, ColumnParallelLinear):
                    op_args['world_size'] = world_size
                    vqlinear = ColumnParallelVQLinear(**op_args)
                elif isinstance(op, RowParallelLinear):
                    op_args['world_size'] = world_size
                    vqlinear = RowParallelVQLinear(**op_args)
                else:
                    raise ValueError(f'Unsupported layer type: {op_name} {op}')
                replace_layer(model, op_name, vqlinear)
        else:
            model.layers[layer_idx] = copy.deepcopy(model.layers[3])
        
    print(f'quantized model: {model}')
    
    # load state dict
    if dry_run is False:
        model_state_dict = load_file(os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        model.load_state_dict(model_state_dict)
    return model


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def interactive_generate(
    model: Transformer,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    world_size: int,
    rank: int
) -> None:
    """
    Handles interactive text generation mode.
    
    Args:
        model (Transformer): The transformer model
        tokenizer: The tokenizer for encoding/decoding text
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for sampling
        world_size (int): Number of distributed processes
        rank (int): Current process rank
    """
    messages = []
    while True:
        if world_size == 1:
            prompt = input(">>> ")
        elif rank == 0:
            prompt = input(">>> ")
            objects = [prompt]
            dist.broadcast_object_list(objects, 0)
        else:
            objects = [None]
            dist.broadcast_object_list(objects, 0)
            prompt = objects[0]
        
        if prompt == "/exit":
            break
        elif prompt == "/clear":
            messages.clear()
            continue
            
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
        completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        print(completion)
        messages.append({"role": "assistant", "content": completion})

def batch_generate(
    model: Transformer,
    tokenizer,
    input_file: str,
    max_new_tokens: int,
    temperature: float,
    max_batch_size: int
) -> None:
    """
    Handles batch text generation mode.
    
    Args:
        model (Transformer): The transformer model
        tokenizer: The tokenizer for encoding/decoding text
        input_file (str): Path to file containing input prompts
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for sampling
        max_batch_size (int): Maximum batch size for generation
    """
    with open(input_file) as f:
        prompts = [line.strip() for line in f.readlines()]
    assert len(prompts) <= max_batch_size
    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
    completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
    completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
    for prompt, completion in zip(prompts, completions):
        print("Prompt:", prompt)
        print("Completion:", completion)
        print()


@torch.inference_mode()
def offload_generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    offload_layers: int = 8
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    
    # Get current process's rank and world size
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = f'cuda:{rank}'  # Each process uses its own GPU
    print(f'offload_generate: rank {rank} using device {device}')
    
    # Start on CPU
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cpu")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cpu")
    prompt_mask = tokens != -1
    
    for cur_pos in range(min(prompt_lens), total_len):
        def _forward(model, tokens, start_pos):
            _tokens = tokens[:, start_pos:cur_pos]
            _start_pos = start_pos
            seqlen = _tokens.size(1)
            
            # move to device
            _tokens = _tokens.to(device)
            model.embed = model.embed.to(device)
            h = model.embed(_tokens)
            model.embed = model.embed.to("cpu")
            
            freqs_cis = model.freqs_cis[_start_pos:_start_pos+seqlen]
            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device="cpu").triu_(1)

            # Process layers in groups of offload_layers size
            total_layers = len(model.layers)
            for start_idx in range(0, total_layers, offload_layers):
                end_idx = min(start_idx + offload_layers, total_layers)
                if rank == 0:
                    print(f'Processing layers {start_idx} to {end_idx-1} on device {device}')
                
                # Move the current group of layers to GPU
                current_layers = model.layers[start_idx:end_idx]
                for layer in current_layers:
                    layer.to(device)
                
                # Move tensors to GPU if needed
                h = h.to(device)
                if freqs_cis is not None:
                    freqs_cis = freqs_cis.to(device)
                if mask is not None:
                    mask = mask.to(device)
                
                # Process all layers in the current group
                for layer_idx, layer in enumerate(current_layers, start=start_idx):
                    if rank == 0:
                        print(f'layer {layer_idx} processing')
                    h = layer(h, _start_pos, freqs_cis, mask)
                
                # Move processed output back to CPU and clear GPU memory
                # h = h.to("cpu")
                for layer in current_layers:
                    layer.to("cpu")

            # Final operations on GPU
            model.norm = model.norm.to(device)
            h = h.to(device)
            h = model.norm(h)[:, -1]
            model.head = model.head.to(device)
            logits = model.head(h)
            
            if world_size > 1:
                all_logits = [torch.empty_like(logits) for _ in range(world_size)]
                dist.all_gather(all_logits, logits)
                logits = torch.cat(all_logits, dim=-1)
            
            model.norm = model.norm.to("cpu")
            model.head = model.head.to("cpu")
            return logits.to("cpu")
        
        logits = _forward(model, tokens, prev_pos)
        
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        
        # move to cpu
        prompt_mask = prompt_mask.to('cpu')
        tokens = tokens.to('cpu')
        next_token = next_token.to('cpu')
        
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens
    

def offload_inference(
    model: Transformer,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    gpu_num: int,
    offload_layers: int
) -> None:
    # print(model)
    rank = torch.cuda.current_device()
    print(f'offload_inference: {rank}')
    
    prompt = "Once upon a time, "
    messages = [{"role": "user", "content": prompt}]
    if rank == 0:
        print(messages)
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    completion_tokens = offload_generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature, offload_layers)
    completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
    
    print(f'--- rank {rank} completion: {completion}')
    
    

def main(
    ckpt_path: str,
    config: str,
    tokenizer_path: str = None,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    offload: bool = False,
    offload_layers: int = 8,
    gpu_num: int = 4,
    dry_run: bool = False,
    quantize: bool = False,
    quant_config: str = ""
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
        offload (bool, optional): Whether to offload the model to CPU. Defaults to False.
        offload_layers (int, optional): Number of layers to offload at a time. Defaults to 8.
    """
    # Always get distributed info
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)  # Set device for NCCL
    
    # Set CPU threads based on mode
    torch.set_num_threads(16)

    global print
    if rank != 0:
        print = lambda *_, **__: None

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(965)
    
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)

    # load tokenizer
    if tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    else:
        print(f'load tokenizer from {tokenizer_path}')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # load model
    start_time = time.time()
    print(f'load model from rank {rank}')
    
    if ckpt_path.endswith(".pt"):
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        with torch.device("cpu"):  # Always load model to CPU first
            model = Transformer(args)
            if quantize is False:
                if not dry_run:
                    # Always load distributed model files
                    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
                    print(f"load_model time: {time.time() - start_time:.2f}s")
                else:
                    print(f"dry run model load")
            else:
                model = get_quantized_deepseek(model, ckpt_path, quant_config, 
                                               world_size=world_size, rank=rank, 
                                               dry_run=dry_run, dtype=torch.bfloat16)
    # generate
    if offload:
        offload_inference(model, tokenizer, max_new_tokens, temperature, gpu_num, offload_layers)
    else:
        if interactive:
            interactive_generate(model, tokenizer, max_new_tokens, temperature, world_size, rank)
        else:
            batch_generate(model, tokenizer, input_file, max_new_tokens, temperature, args.max_batch_size)
    
    # clean up
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--offload-layers", type=int, default=8)
    parser.add_argument("--gpu-num", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quant-config", type=str, default="")
    args = parser.parse_args()
    
    assert args.input_file or args.interactive or args.offload
    main(ckpt_path=args.ckpt_path, 
         tokenizer_path=args.tokenizer_path, 
         config=args.config, 
         input_file=args.input_file, 
         interactive=args.interactive, 
         max_new_tokens=args.max_new_tokens, 
         temperature=args.temperature, 
         offload=args.offload, 
         offload_layers=args.offload_layers,
         gpu_num=args.gpu_num, 
         dry_run=args.dry_run,
         quantize=args.quantize,
         quant_config=args.quant_config)

