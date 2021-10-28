import argparse
import time
import tensorflow as tf
from tensorflow.keras import mixed_precision

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument('--xla', action='store_true')
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()

tf.config.optimizer.set_jit(args.xla)
if args.fp16:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

print("\nConfig:", vars(args))
print("GPU:", tf.config.list_physical_devices("GPU"))
print("")

img_dim = 224
batch_size = args.bs
dataset_size = batch_size*args.steps
classes = 8
example_x = tf.random.uniform(
    (img_dim,img_dim,3), minval=0, maxval=1, dtype=tf.dtypes.float32,
)
example_y = tf.random.uniform(shape=[], minval=0, maxval=classes, dtype=tf.int64)
dataset_x = tf.stack([example_x]*dataset_size)
dataset_y = tf.stack([example_y]*dataset_size)

img_shape = (img_dim,img_dim,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(classes)
inputs = tf.keras.Input(shape=img_shape)
x = base_model(inputs)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

base_learning_rate = 1e-3
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

_ = model.fit(x=dataset_x, y=dataset_y, batch_size=batch_size, epochs=1)

benchmark_epochs = 2
st = time.time()
_ = model.fit(x=dataset_x, y=dataset_y, batch_size=batch_size, epochs=benchmark_epochs)
et = time.time()

fps = round(dataset_size*benchmark_epochs/(et-st), 1)

print("\nResult:")
print("Images/sec training:", fps)
