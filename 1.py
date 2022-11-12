import argparse

my_parser = argparse.ArgumentParser(description=" ")
my_parser.add_argument("--input", metavar="--input", type=str, help="input model")
my_parser.add_argument("--output", metavar="--output", type=str, help="output model")
my_parser.add_argument("--height", metavar="--height", type=int, help="height")
my_parser.add_argument("--width", metavar="--width", type=int, help="width")
args = my_parser.parse_args()

from cain.cain import CAIN
import torch
import os

model = CAIN(3)
model.load_state_dict(torch.load(args.input), strict=False)
input_names = ["input"]
output_names = ["output"]
f1 = torch.rand((1, 3, args.height, args.width * 2))
x = f1

torch.onnx.export(
    model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    "cain-temp.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=15,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=input_names,  # the model's input names
    output_names=output_names,
)  #                  dynamic_axes={'input' : {3 : 'width', 2: 'height'}})
del model
os.system("python3 -m onnxsim cain-temp.onnx cain-sim.onnx")
os.system(
    f"polygraphy convert cain-sim.onnx --fp16 --convert-to trt  --workspace 10737418240 -o {args.output}"
)
