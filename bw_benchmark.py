import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default=100, type=int,
                    help="Number of iterations to run within each benchmark")
parser.add_argument("--device1", default="/GPU:0", type=str)
parser.add_argument("--device2", default="/GPU:0", type=str)
args = parser.parse_args()

import time
import tensorflow as tf

#tf.config.set_visible_devices([], 'GPU')

# set this to at least the number of execution units for optimal measurement
# always set it as a integer multiple to avoid tile/wave quantization effect
D = 8192

op_type = "mult-add"

input_shape = (D,D,)
num_floats = D*D

op_flops = {
    "add": num_floats,
    "mult": num_floats,
    "mult-add": 2*num_floats
}

flops = op_flops[op_type]

# note: 32b (bit) = 4B (byte)
# memory estimation:
#  - add/mult
# 4B + 4B -> 4B 
# total: 8B read + 4B write = 12B
# - mult-add
# 2x 4B + 4B -> 4B 
# total: 16B read + 8B write = 24B

op_memory = {
    "add": 12*num_floats,
    "mult": 12*num_floats,
    "mult-add": 24*num_floats
}

# 4 arrays of D*D * 4 bytes (32bit) * 2 access
mem_GB = op_memory[op_type]/(1e9)

print("")

print("Est. Memory:", round(mem_GB,1), "GB")
print("      FLOPs:", flops)
print("")

with tf.device(args.device1):
    a = tf.random.normal(input_shape, dtype=tf.dtypes.float32)
    b = tf.random.normal(input_shape, dtype=tf.dtypes.float32)
    c = tf.random.normal(input_shape, dtype=tf.dtypes.float32)
    
@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def run_mult_add(a, b, c):
    print("Tracing run_mult_add")
    x = a*b+c
    return x

with tf.device(args.device2):
    C = run_mult_add(a, b, c)

    print("\nStarting device:", a.device)
    print("Ending device:", C.device)

    C.numpy()
    C = run_mult_add(a, b, c)
    C.numpy()

    # measure overhead
    st = time.time()
    for i in range(30):
        for j in range(1):
            C = run_mult_add(a, b, c)
        C.numpy()
    et = time.time()
    overhead = (et-st)/30

    print("\nOverhead:", overhead)
    print("")

    try:
        while True:
            st = time.time()
            for i in range(args.iterations):
                C = run_mult_add(a, b, c)
            C.numpy()
            et = time.time()
            duration = et-st-overhead
            fps = args.iterations/duration
            gflops = fps*flops/(1e9)
            bw = fps*mem_GB
            print("it/sec:", round(fps,1), "GFLOPS:", round(gflops,1), "est.BW:", round(bw,1), "GB/s")
    except KeyboardInterrupt:
        pass
