import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default=30, type=int,
                    help="Number of iterations to run within each benchmark")
parser.add_argument("--device1", default="/CPU:0", type=str)
parser.add_argument("--device2", default="/GPU:0", type=str)
args = parser.parse_args()

import os
import time
from tqdm import tqdm
import tensorflow as tf

@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def do_op(a, b):
    with tf.device(args.device1):
        x = a * b + b
    with tf.device(args.device2):
        x = tf.linalg.matmul(a, x)
    with tf.device(args.device1):
        x = a * x + b
    with tf.device(args.device2):
        x = tf.linalg.matmul(b, x)
    with tf.device(args.device1):
        x = a * b + x
    with tf.device(args.device2):
        x = tf.linalg.matmul(a, x)
    with tf.device(args.device1):
        x = a * b + x
    with tf.device(args.device2):
        x = tf.linalg.matmul(b, x)
    return x

def benchmark_matmul(M, dtype=tf.float32, iterations=30):
    # generate data and warm-up iteration
    A = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
    B = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
    C = do_op(A, B)
    C.numpy()
    C = do_op(A, B)
    C.numpy()
    # run benchmark
    st = time.time()
    for _ in range(iterations+1):
        C = do_op(A, B)
    C.numpy()
    et = time.time()
    duration = (et-st)
    return iterations/duration

fp16_matmul, fp32_matmul, fp64_matmul = [], [], []
fp16_tflops, fp32_tflops, fp64_tflops = [], [], []

M_list = [2048] * 30

print("\nStarting burn...\n")

burn_start = time.time()

for M in tqdm(M_list):
    print("FP32", M, end=" : ")
    ret = benchmark_matmul(M, dtype=tf.float32, iterations=args.iterations)
    tflops = 4 * (ret * 2 * M**3 + 2*M*M)/ 1e12
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
print("* FP32:", round(max(fp32_tflops),1), "TFLOPS")
print("")

