import argparse
import os
import torch
from torch import nn
from deepseek.kernel import weight_dequant
from deepseek.model import ColumnParallelLinear, RowParallelLinear, Linear

# load files from a folder
def load_files(input_dir):
    files = os.listdir(input_dir)
    return files

# find specific layers in a model
def find_layers(module, target_layers=[nn.Linear], name=''):
    if type(module) in target_layers:
        return {name: module}
    res = {}
    for old_name, child in module.named_children():
        res.update(find_layers(child, target_layers=target_layers, name=name + '.' + old_name if name != '' else old_name))
    return res

def convert_layer(layer):
    target_layers = [ColumnParallelLinear, RowParallelLinear, Linear]
    ops = find_layers(layer, target_layers)
    print(f'find operations: {ops}')
    for name, op in ops.items():
        if op.weight.dtype == torch.float8_e4m3fn:
            bfloat16_weight = weight_dequant(x=op.weight, s=op.scale, block_size=128)
            bfloat16_weight = bfloat16_weight.to(torch.bfloat16)
            op.weight = torch.nn.Parameter(bfloat16_weight, requires_grad=False)
    return layer

def convert_layers(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = load_files(input_dir)
    for file in files:
        file_path = os.path.join(input_dir, file)
        print(f'loading {file_path}')
        layer = torch.load(file_path, weights_only=False)
        layer = layer.to('cuda')
        converted_layer = convert_layer(layer)
        converted_layer = converted_layer.to('cpu')
        print(f'converted layer: {converted_layer}')
        torch.save(converted_layer, os.path.join(output_dir, file))
        print(f'saving {file_path}')
        del layer
        del converted_layer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()
    convert_layers(args.input_dir, args.output_dir)


