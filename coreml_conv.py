import time
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import tensorflow as tf

MN = 3
CK = 256
HW = 512
HW_og = 512
num_conv = 50
batch_size = 1

model = tf.keras.Sequential(
    layers=[
        tf.keras.Input(shape=(HW, HW, 3,)),
    ],
    name="model",
)

model_flops = 0

model.add(tf.keras.layers.Conv2D(filters=CK,
                               kernel_size=(MN,MN), 
                               activation=None, 
                               use_bias=False))
conv_flops = MN * MN * CK * 3 * HW * HW

HW = int((HW - MN + 0) + 1)

for i in range(num_conv-1):
    model.add(tf.keras.layers.Conv2D(filters=CK,
                               kernel_size=(MN,MN), 
                               activation=None, 
                               use_bias=False))
    HW = int((HW - MN + 0) + 1)
    conv_flops = MN * MN * CK * CK * HW * HW
    model_flops += conv_flops

print("")
print("FLOPs:", model_flops)
print("")

tf_out = model.predict(np.random.rand(batch_size,HW_og,HW_og,3))
print("tf_out", tf_out.shape)

model.summary()

print("Check HW:", HW)

print(model.inputs)

print("Converting model")

mlmodel = ct.convert(model,
                     inputs=[ct.TensorType(shape=(batch_size,HW_og,HW_og,3))],
                     compute_units=ct.ComputeUnit.ALL)

mlmodel = quantization_utils.quantize_weights(mlmodel, 16)
                             
print("Testing model")

random_input = {"input_1": np.random.rand(batch_size,HW_og,HW_og,3)}

ane_out = mlmodel.predict(random_input)

print("ane_out", ane_out["Identity"].shape)

print("Benchmarking model")

# warmup

ane_out = mlmodel.predict(random_input)

iterations = 30

st = time.time()
for i in range(iterations):
    ane_out = mlmodel.predict(random_input)
et = time.time()
    
print("ane_out", ane_out["Identity"].shape)

duration = et-st
fps = batch_size*iterations/duration
tfops = fps * model_flops / 1e12

print("TFLOPS", tfops)

