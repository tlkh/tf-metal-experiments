import time
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import tensorflow as tf

num_matmul = 10
D = 1024*4

class MatMul(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MatMul, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        print("w.shape", self.w.shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

model = tf.keras.Sequential(
    layers=[
        tf.keras.Input(shape=(D,)),
    ],
    name="model",
)

for i in range(num_matmul):
    model.add(MatMul(D))

batch_size = 1

model.build([batch_size, D])

tf_out = model.predict(np.random.rand(batch_size,D,D,))
print("tf_out", tf_out.shape)

model.summary()

print(model.inputs)

print("Converting model")

mlmodel = ct.convert(model,
                     convert_to="mlprogram",
                     inputs=[ct.TensorType(shape=(batch_size,D,D))],
                     compute_precision=ct.precision.FLOAT16,
                     compute_units=ct.ComputeUnit.ALL)

#mlmodel = quantization_utils.quantize_weights(mlmodel, 16)
                             
print("Testing model")

random_input = {"input_1": np.random.rand(batch_size,D,D,)}

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
fps = batch_size*num_matmul*iterations/duration
tops = (fps * 2 * D**3) / 1e12

print("tops", tops)

