import tensorflow as tf
try:
    from transformers import TFAutoModelForSequenceClassification
except Exception as e:
    print(e)

def return_transformer(model_name, seq_len=128, batch_size=32, steps=30, num_labels=8):
    # data
    dataset_size = batch_size*steps
    example_x = tf.random.uniform(
        (seq_len,), minval=0, maxval=1000, dtype=tf.dtypes.int64,)
    example_y = tf.random.uniform(shape=[], minval=0, maxval=num_labels, dtype=tf.int64)
    dataset_x = tf.stack([example_x]*dataset_size)
    dataset_y = tf.stack([example_y]*dataset_size)
    # model
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="../cache/")
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    return {"sample_data": [dataset_x, dataset_y],
            "model": model,}

def return_cnn(model_name, img_dim=224, batch_size=32, steps=30, num_labels=8):
    # data
    dataset_size = batch_size*steps
    example_x = tf.random.uniform(
        (img_dim,img_dim,3), minval=0, maxval=1, dtype=tf.dtypes.float32,)
    example_y = tf.random.uniform(shape=[], minval=0, maxval=num_labels, dtype=tf.int64)
    dataset_x = tf.stack([example_x]*dataset_size)
    dataset_y = tf.stack([example_y]*dataset_size)
    # model
    img_shape = (img_dim,img_dim,3)
    include_top = False
    if model_name == "resnet50":
        base_model = tf.keras.applications.resnet50.ResNet50(input_shape=img_shape, include_top=include_top)
    elif model_name == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=include_top)
    else:
        raise ValueError("Model not supported")
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_labels)
    inputs = tf.keras.Input(shape=img_shape)
    x = base_model(inputs)
    x = global_average_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),)
    return {"sample_data": [dataset_x, dataset_y],
            "model": model,}
