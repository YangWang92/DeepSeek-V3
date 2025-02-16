import torch
from deepseek.model import Transformer, ModelArgs
from safetensors.torch import load_model
import argparse
import os
import json

if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("--model_path", type=str, default="model0-mp1.safetensors")
	args.add_argument("--config", type=str, default="config.json")
	args.add_argument("--output_dir", type=str, default="layers")
	args = args.parse_args() 
	os.makedirs(args.output_dir, exist_ok=True)
	
	# load config
	with open(args.config) as f:
		model_args = ModelArgs(**json.load(f))
	print(f'model_args: {model_args}')
    	
	# load model
	print(f'Loading model from {args.model_path}...')
	model = Transformer(model_args)
	load_model(model, args.model_path)
 
	n_layers = len(model.layers)
	# save layers
	for layer_idx in range(n_layers):
		print(f"Saving layer {layer_idx}...")
		torch.save(model.layers[layer_idx], os.path.join(args.output_dir, f"layer_{layer_idx}.pt"))

	print("Done!")
