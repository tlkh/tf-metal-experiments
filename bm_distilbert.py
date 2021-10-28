import argparse
import time
import tensorflow as tf
from tensorflow.keras import mixed_precision
from transformers import TFAutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=64)
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

seq_len = 128
batch_size = args.bs
dataset_size = batch_size*args.steps
classes = 8
example_x = tf.random.uniform(
    (seq_len,), minval=0, maxval=1000, dtype=tf.dtypes.int64,
)
example_y = tf.random.uniform(shape=[], minval=0, maxval=classes, dtype=tf.int64)
dataset_x = tf.stack([example_x]*dataset_size)
dataset_y = tf.stack([example_y]*dataset_size)

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=8)
model.summary()

base_learning_rate = 1e-3
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

_ = model.fit(x=dataset_x, y=dataset_y, batch_size=batch_size, epochs=1)

benchmark_epochs = 2
st = time.time()
_ = model.fit(x=dataset_x, y=dataset_y, batch_size=batch_size, epochs=benchmark_epochs)
et = time.time()

fps = round(dataset_size*benchmark_epochs/(et-st), 1)

print("\nResult:")
print("Seq/sec training:", fps)
