import time
import tensorflow as tf
import math

#tf.config.set_visible_devices([], 'GPU')

MN = 3
CK = 128
HW = 256
batch_size = 32
conv_type = "normal"

if conv_type=="normal":
    conv_layer = tf.keras.layers.Conv2D(
        filters=CK, 
        kernel_size=(MN,MN), 
        activation=None, 
        use_bias=False
    )
    conv_flops = MN * MN * CK * CK * HW * HW
    conv_params = MN * MN * CK * CK
elif conv_type=="depsep":
    conv_layer = tf.keras.layers.SeparableConv2D(
        filters=CK, 
        kernel_size=(MN,MN), 
        activation=None, 
        use_bias=False
    )
    conv_flops = MN * MN * CK * HW * HW + CK * CK * HW * HW
    conv_params = MN * MN * CK + CK * CK
    
print("")
print("Conv2D type:", conv_type)
print("Params:", conv_params)
print("FLOPs:", conv_flops)
print("")

input_shape = (batch_size, HW, HW, CK)
in_data = tf.random.normal(input_shape)

@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def run_conv(in_data):
    print("Tracing")
    return conv_layer(in_data)

iterations = 30

C = run_conv(in_data)
C.numpy()

print("Start benchmark")

try:
    while True:
        st = time.time()
        for i in range(iterations):
            C = run_conv(in_data)
        C.numpy()
        et = time.time()
        duration = et-st
        fps = batch_size*iterations/duration
        tflops = fps*conv_flops/(1e12)
        print("Conv/sec:", round(fps,1), "TFLOPS:", round(tflops,1))
except KeyboardInterrupt:
    pass




    