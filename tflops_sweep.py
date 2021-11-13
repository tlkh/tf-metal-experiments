import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default=5, type=int,
                    help="Number of iterations to run within each benchmark")
args = parser.parse_args()

import os
import time
from tqdm import tqdm
import tensorflow as tf

@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def do_op(a, b, num_matmul=10):
    print("Tracing")
    x = tf.linalg.matmul(a, b)
    for _ in range(num_matmul-1):
        x = tf.linalg.matmul(a, x)
    return x

def benchmark_matmul(M, dtype=tf.float32, num_matmul=100, iterations=1):
    # generate data
    with tf.device("/GPU:0"):
        A = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
        B = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
    # warm-up iteration
    C = do_op(A, B, num_matmul=num_matmul)
    C.numpy()
    C = do_op(A, B, num_matmul=num_matmul)
    C.numpy()
    time.sleep(1)
    # run benchmark
    st = time.time()
    for _ in range(iterations):
        C = do_op(A, B, num_matmul=num_matmul)
    C.numpy()
    et = time.time()
    duration = et-st
    return num_matmul*iterations/duration

fp32_tflops = []

M_list = [32, 64, 128, 256, 512, 1024, 1536, 2048, 4096, 6144, 8192]

print("\nStarting burn...\n")

burn_start = time.time()

for M in tqdm(M_list):
    print("FP32", M, end=" : ")
    fps = benchmark_matmul(M, dtype=tf.float32, iterations=args.iterations)
    tflops = fps * 2 * M**3 / 1e12
    fp32_tflops.append(tflops)
    print(tflops)
    
burn_end = time.time()
    
print("\nFinished in", int(burn_end-burn_start), "seconds\n")

max_tflop = max(fp32_tflops)
max_tflop_M = M_list[fp32_tflops.index(max_tflop)]
    
title = "Max TFLOPS achieved"
print("")
print(title)
print("="*len(title))
print("* FP32:", round(max_tflop, 1), "TFLOPS")
print("")

from matplotlib import pyplot as plt
plt.clf()
plt.figure(figsize=(10,6), dpi=100)
plt.title(title)
plt.plot(M_list, fp32_tflops, label="FP32", color="b")
plt.axvline(max_tflop_M, color="k", linestyle="--", linewidth=1, label="M="+str(max_tflop_M))
plt.xlabel("Matrix size M*M")
plt.ylabel("Achieved TFLOPS")
plt.legend()
plt.savefig("gpu_tflops_plot.jpg")
