import time
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))

img_dim = 224
batch_size = 128
dataset_size = batch_size*30
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
