import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default=30, type=int,
                    help="Number of iterations to run within each benchmark")
args = parser.parse_args()

import os
import time
from tqdm import tqdm
import tensorflow as tf

@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def do_op(a, b):
    return tf.linalg.matmul(a, b)

def benchmark_matmul(M, dtype=tf.float32, iterations=30):
    # generate data and warm-up iteration
    with tf.device("/GPU:0"):
        A = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
        B = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
        C = do_op(A, B)
    C.numpy()
    # run benchmark
    st = time.time()
    with tf.device("/GPU:0"):
        for _ in range(iterations+1):
            C = do_op(A, B)
    C.numpy()
    et = time.time()
    duration = (et-st)
    return iterations/duration

fp16_matmul, fp32_matmul, fp64_matmul = [], [], []
fp16_tflops, fp32_tflops, fp64_tflops = [], [], []

M_list = [8192] * 30

print("\nStarting burn...\n")

burn_start = time.time()

for M in tqdm(M_list):
    print("FP32", M, end=" : ")
    ret = benchmark_matmul(M, dtype=tf.float32, iterations=args.iterations)
    tflops = ret * 2 * M**3 / 1e12
    fp32_matmul.append(ret)
    fp32_tflops.append(tflops)
    print(tflops)
    #time.sleep(1)
    
burn_end = time.time()
    
print("\nFinished in", int(burn_end-burn_start), "seconds\n")
    
title = "Max TFLOPS achieved"
print("")
print(title)
print("="*len(title))
print("* FP32:", int(max(fp32_tflops)), "TFLOPS")
print("")

