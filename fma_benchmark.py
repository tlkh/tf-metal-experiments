import time
import tensorflow as tf
import math

tf.config.set_visible_devices([], 'GPU')

D = 1024*20

input_shape = (D,D,)

flops = 2*D*D
mem = 4*D*D*4*2/(1e9)

print("")
# FYI probably not very accurate lol
print("Est. Memory:", round(mem,1), "GB")
print("      FLOPs:", flops)
print("")

input("Press Enter to continue...")

with tf.device("/GPU:0"):
    a = tf.random.normal(input_shape)
    b = tf.random.normal(input_shape)
    c = tf.random.normal(input_shape)
    
@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def run_fma(a, b, c):
    print("Tracing")
    return a*b+c

iterations = 1000

C = run_fma(a, b, c)
C.numpy()

print("Confirm your system is NOT going to swap before continue")
print("I don't this will be healthy for any SSD")
input("Press Enter to continue...")

print("Start benchmark")

try:
    while True:
        st = time.time()
        for i in range(iterations):
            C = run_fma(a, b, c)
        C.numpy()
        et = time.time()
        print(C.shape)
        duration = et-st
        fps = iterations/duration
        tflops = fps*flops/(1e9)
        print("it/sec:", round(fps,1), "GFLOPS:", round(tflops,1))
except KeyboardInterrupt:
    pass




    