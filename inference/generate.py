import os
import json
from argparse import ArgumentParser
from typing import List

from sympy import threaded
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model, safe_open

from model import Transformer, ModelArgs

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


def offload_generate(
    model: Transformer,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    gpu_num: int
) -> None:
    print(f"offload_generate: {gpu_num}")
    print(model)
    
    
def parallel_load_model(model, ckpt_path, cpu_threads=32):
    """
    Load model weights in parallel using multiple threads with optimized performance.
    """
    import threading
    from queue import Queue
    import os
    from safetensors.torch import safe_open
    import time
    
    start_time = time.time()
    files = sorted([f for f in os.listdir(ckpt_path) if f.startswith('model-') and f.endswith('.safetensors')])
    print(f"Found {len(files)} weight files")
    
    file_queue = Queue()
    result_queue = Queue()
    
    def worker(thread_id):
        while True:
            try:
                filepath = os.path.join(ckpt_path, files[thread_id])
                local_dict = {}
                print(f"Thread {thread_id}: Loading file {filepath}")
                with safe_open(filepath, framework="pt") as f:
                    keys = list(f.keys())
                    
                    for k in keys:
                        tensor = f.get_tensor(k).clone().to('cpu')
                        local_dict[".".join(k.split('.')[1:])] = tensor
                
                result_queue.put((thread_id, local_dict))
                # print(f"Thread {thread_id}: Loaded file {files[thread_id]}")
            except Exception as e:
                print(f"Error in thread {thread_id} loading {files[thread_id]}: {e}")
                result_queue.put((thread_id, None))
    
    for i in range(len(files)):
        file_queue.put(i)
    
    threads = []
    num_threads = min(cpu_threads, len(files))
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
    
    results = {}
    total_size = 0
    completed = 0
    total_files = len(files)
    
    while completed < total_files:
        idx, tensors = result_queue.get()
        if tensors is not None:
            results.update(tensors)
            for tensor in tensors.values():
                total_size += tensor.numel() * tensor.element_size()
            completed += 1
            print(f"Progress: {completed}/{total_files} files loaded. "
                  f"Current size: {total_size / 1024**3:.2f} GB. "
                  f"Time elapsed: {time.time() - start_time:.2f}s")
    
    for t in threads:
        t.join()
    
    print(f"\nLoading completed in {time.time() - start_time:.2f} seconds")
    print(f"Total loaded weights: {len(results)}")
    print(f"Total size: {total_size / 1024**3:.2f} GB")
    
    print(results.keys())
    
    try:
        model.load_state_dict(results, strict=True)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading weights into model: {e}")
        print('--------------------------------')
        print(model.state_dict().keys())
        raise

    return model

def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    offload: bool = False,
    gpu_num: int = 4,
    cpu_threads: int = 48
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
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    if offload is False:
        torch.cuda.set_device(local_rank)
    else:
        torch.set_default_device("cpu")

    torch.set_default_dtype(torch.bfloat16)
    if offload is False:
        torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda" if offload is False else "cpu"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    # load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    import time
    start_time = time.time()
    if offload:
        parallel_load_model(model, ckpt_path, cpu_threads=cpu_threads)
    else:
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    print(f"load_model time: {time.time() - start_time:.2f}s")
    
    # import time
    # start_time = time.time()
    # load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    # print(f"load_model time: {time.time() - start_time:.2f}s")
    
    if offload is False:
        if interactive:
            interactive_generate(model, tokenizer, max_new_tokens, temperature, world_size, rank)
        else:
            batch_generate(model, tokenizer, input_file, max_new_tokens, temperature, args.max_batch_size)
    else:
        offload_generate(model, tokenizer, max_new_tokens, temperature, gpu_num)

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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--gpu-num", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    assert args.input_file or args.interactive or args.offload
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.offload, args.gpu_num, args.cpu_threads)
