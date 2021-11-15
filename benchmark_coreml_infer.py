import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--type", type=str, default="cnn")
parser.add_argument("--bs", type=int, default=1)
args = parser.parse_args()

import time
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import tensorflow as tf
from model_library import *

if args.type == "cnn":
    train_items = return_cnn(args.model, batch_size=args.bs, steps=1)
elif args.type == "transformer":
    train_items = return_transformer(args.model, batch_size=args.bs, steps=1)
else:
    raise ValueError("Model type not supported")

model = train_items["model"]
dataset_x, dataset_y = train_items["sample_data"]

sample_input = dataset_x
input_size = list(sample_input.shape)
print("input_size", input_size)
    
tf_out = model.predict(sample_input)

model.summary()

mlmodel = ct.convert(model,
                     convert_to="mlprogram",
                     inputs=[ct.TensorType(shape=input_size)],
                     compute_precision=ct.precision.FLOAT16,
                     compute_units=ct.ComputeUnit.CPU_AND_GPU)

random_input = {"input_2": np.asarray(sample_input).astype(float)}

ane_out = mlmodel.predict(random_input)

# warmup

ane_out = mlmodel.predict(random_input)

iterations = 100

st = time.time()
for i in range(iterations):
    ane_out = mlmodel.predict(random_input)
et = time.time()

duration = et-st
fps = args.bs*iterations/duration

print("\nSample/sec", fps)
print("")

