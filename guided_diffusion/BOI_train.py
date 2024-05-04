"""
Train a diffusion model on images.
"""
import os
import datetime
import numpy as np
import pickle
import argparse
import blobfile as bf
from guided_diffusion import dist_util, logger
from guided_diffusion.BOI_image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def load_clusters(cluster_file, idx):
    """
    Load clusters of images.

    cluster_file: .pkl file containing a dictionary mapping cluster indices to
    lists of image paths.
    """
    clusters = {}
    with open(cluster_file, "rb") as f:
        clusters = pickle.load(f)

    assert idx in clusters, f"Cluster {idx} not found in {cluster_file}"

    return clusters[idx]


def main():
    
    args = create_argparser().parse_args()
    if not args.cluster_file:
        raise ValueError("unspecified cluster data directory")

    dist_util.setup_dist()
    if args.out_dir is not None:
        logger.configure(dir=os.path.join(args.out_dir, f"cluster{args.cluster_index}", datetime.datetime.now().strftime("diffgen-%Y-%m-%d-%H-%M")))
    else:
        logger.configure()

    logger.log("creating model and diffusion...")

    args.in_channels = 4

    logger.log(f"Loading cluster {args.cluster_index}...")
    cluster = load_clusters(args.cluster_file, args.cluster_index)
    # Note that cluster is a list of paths to images
    cluster_size = len(cluster)

    logger.log(f"Cluster size: {cluster_size}")
    adjusted_cluster = list(map(lambda x: bf.join(f"{args.data_dir}/masked-images/", bf.basename(x)), cluster))

    # Create a model and diffusion object and dataloader for each image cluster
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    # Path of the form: data_dir/masked-images/filename.ext

    logger.log("creating data loader...")
    data = load_data(
        img_paths=adjusted_cluster,
        batch_size=args.batch_size if args.batch_size < cluster_size else cluster_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    # Train the models in parallel
    logger.log(f"training on cluster_{args.cluster_index}...")
    logger.log(f"Saving every {args.save_interval} steps with a maximum of {args.max_steps} steps.")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        save_prefix=f"cluster_{args.cluster_index}_",
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop(max_steps=args.max_steps)



def create_argparser():
    # In this case, resume_checkpoint is the path to the dir + /model{step}.pt
    # image_size 256
    defaults = model_and_diffusion_defaults()
    # The settings below will be overwritten by any command line arguments
    defaults.update(dict(
        data_dir='',
        out_dir=None,
        cluster_index=0,
        cluster_file="/root/diffusion-gen/clusters.pkl",
        schedule_sampler="uniform",
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=128,
        microbatch=4,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=1000,
        max_steps=3600,    # some integer multiple of save_interval
        resume_checkpoint="",
        fp16_scale_growth=1e-3,
    ))
    # Diffusion model settings
    defaults.update(dict(
        attention_resolutions="16",
        class_cond=False,
        diffusion_steps=1000,
        rescale_timesteps=True,
        learn_sigma=True,
        num_channels=256,
        noise_schedule="linear",
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
