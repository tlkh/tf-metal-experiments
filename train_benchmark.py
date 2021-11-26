import argparse
import time
import wandb
import tensorflow as tf
from tensorflow.keras import mixed_precision
from model_library import *

PROJECT = "tf-metal-experiments"
ENTITY = None  #replace with the team id

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--type", type=str, default="cnn")
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--xla", action="store_true")
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()

print(args)

with wandb.init(project=PROJECT, entity=ENTITY, config=args.__dict__):
    tf.config.optimizer.set_jit(args.xla)
    if args.fp16:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        
    # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0],
    #                                         True)

    print("\nConfig:", vars(args))
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("")

    print("\n[1/3] Loading model\n")

    if args.type == "cnn":
        train_items = return_cnn(args.model, batch_size=args.bs, steps=args.steps)
    elif args.type == "transformer":
        train_items = return_transformer(args.model, batch_size=args.bs, steps=args.steps)
    else:
        raise ValueError("Model type not supported")
        
    model = train_items["model"]
    dataset_x, dataset_y = train_items["sample_data"]

    model.summary()

    print("\n[2/3] Warm-up run\n")

    _ = model.fit(x=dataset_x, y=dataset_y, batch_size=args.bs, epochs=1, verbose=1)

    print("\n[3/3] Benchmark run\n")
    cbs = [wandb.keras.WandbCallback(save_model=False),]
    st = time.time()
    _ = model.fit(x=dataset_x, y=dataset_y, batch_size=args.bs, callbacks=cbs, epochs=args.epochs, verbose=2)
    et = time.time()

    dataset_size = args.bs*args.steps
    fps = round(dataset_size*args.epochs/(et-st), 1)

    print("\nTraining Result:")
    print("Sample/sec:", fps)
    wandb.log({"fps":fps})