import time
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

print(tf.config.list_physical_devices("GPU"))

seq_len = 128
batch_size = 64
dataset_size = batch_size*10
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
