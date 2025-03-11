from email import generator
import os
import json
from argparse import ArgumentParser
from sys import argv
from typing import List, Generator
import copy

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model, safe_open, load_file
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
                           dtype=torch.bfloat16):
    num_layers = len(model.layers)
    layers = model.layers
    quant_config = get_layer_config(quant_config)

    target_layers = [ColumnParallelLinear, RowParallelLinear, Linear]
    
     
    for layer_idx in tqdm(range(num_layers), desc="Initializing"):
        # hack, all layers are the same vector length and num centroids
        ops = find_layers(layers[layer_idx], target_layers)
        for op_name, op in ops.items():
            op_name = f'layers.{layer_idx}.{op_name}'
            op_args = quant_config[op_name]
            op_args = convert_str_to_dtypes(op_args)
            # drop norm_dim to workaround the config 
            op_args.pop('norm_dim', None)
            if type(op) == ColumnParallelLinear:
                vqlinear = ColumnParallelVQLinear(**op_args)
            elif type(op) == RowParallelLinear:
                vqlinear = RowParallelVQLinear(**op_args)
            elif type(op) == Linear:
                vqlinear = VQuantLinear(**op_args)
            else:
                raise ValueError(f'Unsupported layer type: {op_name} {op}')
            replace_layer(model, op_name, vqlinear)
        # if layer_idx <= 3:
        #     ops = find_layers(layers[layer_idx], target_layers)
        #     for op_name, op in ops.items():
        #         op_name = f'layers.{layer_idx}.{op_name}'
        #         op_args = quant_config[op_name]
        #         op_args = convert_str_to_dtypes(op_args)
        #         # drop norm_dim to workaround the config 
        #         op_args.pop('norm_dim', None)
        #         if type(op) == ColumnParallelLinear:
        #             vqlinear = ColumnParallelVQLinear(**op_args)
        #         elif type(op) == RowParallelLinear:
        #             vqlinear = RowParallelVQLinear(**op_args)
        #         elif type(op) == Linear:
        #             vqlinear = VQuantLinear(**op_args)
        #         else:
        #             raise ValueError(f'Unsupported layer type: {op_name} {op}')
        #         replace_layer(model, op_name, vqlinear)
        # else:
        #     model.layers[layer_idx] = copy.deepcopy(model.layers[3])
    
    if rank == 0:    
        print(f'quantized model: {model}')
    
    # load state dict
    model_state_dict = load_file(os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    model.to('cuda')
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
def _interactive_generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    stream: bool = False
) -> List[List[int]] | Generator[int, None, None]:
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
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
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
        
        if stream:
            yield next_token.item()
            
        if finished.all():
            break

    if not stream:
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
    Handles interactive text generation mode with streaming output.
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
        
        token_buffer = []
        for token in _interactive_generate(model, [prompt_tokens], max_new_tokens, 
                            tokenizer.eos_token_id, temperature, stream=True):
            token_buffer.append(token)
            if len(token_buffer) >= 1:
                text = tokenizer.decode(token_buffer, skip_special_tokens=True)
                print(text, end="", flush=True)
                token_buffer = []
                
        if token_buffer:
            text = tokenizer.decode(token_buffer, skip_special_tokens=True)
            print(text, end="", flush=True)
        print() 
        
        completion = tokenizer.decode(token_buffer, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": completion})

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
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
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


def batch_generate(
    model: Transformer,
    tokenizer,
    input_file: str,
    max_new_tokens: int,
    temperature: float,
    max_batch_size: int
) -> None:
    with open(input_file) as f:
        prompts = [line.strip() for line in f.readlines()]
    assert len(prompts) <= max_batch_size, f"Number of prompts exceeds maximum batch size ({max_batch_size})"
    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
    completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
    completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
    for prompt, completion in zip(prompts, completions):
        print("Prompt:", prompt)
        print("Completion:", completion)
        print()
    with open("output.txt", "w") as f:
        for prompt, completion in zip(prompts, completions):
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Completion: {completion}\n")
            f.write("\n")

def main(
    ckpt_path: str,
    config: str,
    tokenizer_path: str = None,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
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
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
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
    model = Transformer(args)
    if quantize is False:
        # Always load distributed model files
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    else:
        model = get_quantized_deepseek(model, ckpt_path, quant_config, 
                                       world_size=world_size, rank=rank, 
                                       dtype=torch.bfloat16)
    
    if interactive:
        interactive_generate(model, tokenizer, max_new_tokens, temperature, world_size, rank)
    else:
        batch_generate(model, tokenizer, input_file, max_new_tokens, temperature, args.max_batch_size)
    
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
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quant-config", type=str, default="")
    args = parser.parse_args()
     
    assert args.input_file or args.interactive
    
    os.environ['NCCL_DEBUG'] = 'NONE'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'NONE'
    os.environ['TORCH_DISTRIBUTED_TIMEOUT'] = '3600'
    os.environ['NCCL_TIMEOUT'] = '3600'
    os.environ['NCCL_SOCKET_TIMEOUT'] = '360000'
    
    main(ckpt_path=args.ckpt_path, 
         tokenizer_path=args.tokenizer_path, 
         config=args.config, 
         input_file=args.input_file, 
         interactive=args.interactive, 
         max_new_tokens=args.max_new_tokens, 
         temperature=args.temperature, 
         quantize=args.quantize,
         quant_config=args.quant_config)
