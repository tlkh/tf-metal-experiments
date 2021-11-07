import time
import tensorflow as tf

#tf.config.set_visible_devices([], 'GPU')

# set this to at least the number of execution units for optimal measurement
# always set it as a integer multiple to avoid tile/wave quantization effect
D = 8192

op_type = "fma"

input_shape = (D,D,)
num_floats = D*D

op_flops = {
    "add": num_floats,
    "mult": num_floats,
    "fma": 2*num_floats
}

flops = op_flops[op_type]

# note: 32b (bit) = 4B (byte)
# memory estimation:
#  - add/mult
# 4B + 4B -> 4B 
# total: 8B read + 4B write = 12B
# - fma
# 2x 4B + 4B -> 4B 
# total: 16B read + 8B write = 24B

op_memory = {
    "add": 12*num_floats,
    "mult": 12*num_floats,
    "fma": 24*num_floats
}

# 4 arrays of D*D * 4 bytes (32bit) * 2 access
mem_GB = op_memory[op_type]/(1e9)

print("")

print("Est. Memory:", round(mem_GB,1), "GB")
print("      FLOPs:", flops)
print("")

input("Press Enter to continue...")

with tf.device("/GPU:0"):
    a = tf.random.normal(input_shape)
    b = tf.random.normal(input_shape)
    c = tf.random.normal(input_shape)
    
@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def run_fma(a, b, c):
    print("Tracing run_fma(a, b, c)")
    return a*b+c

C = run_fma(a, b, c)
C = run_fma(a, b, c)
C.numpy()

# measure overhead
st = time.time()
for i in range(30):
    for j in range(1):
        C = run_fma(a, b, c)
    C.numpy()
et = time.time()
overhead = (et-st)/30

print("Overhead:", overhead)
print(" ")
print("Confirm your system is NOT going to swap before continue")
print(" ")
input("Press Enter to continue...")
print("Start benchmark")

iterations = 1000

try:
    while True:
        st = time.time()
        for i in range(iterations+1):
            C = run_fma(a, b, c)
        C.numpy()
        et = time.time()
        duration = et-st-overhead
        fps = iterations/duration
        gflops = fps*flops/(1e9)
        bw = fps*mem_GB
        print("it/sec:", round(fps,1), "GFLOPS:", round(gflops,1), "est BW:", round(bw,1), "GB/s")
except KeyboardInterrupt:
    pass

