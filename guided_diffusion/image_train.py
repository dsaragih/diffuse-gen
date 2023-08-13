"""
Train a diffusion model on images.
"""
import os
import datetime
import cv2
import numpy as np
import argparse
# import blobfile as bf
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    if args.out_dir is not None:
        logger.configure(dir=os.path.join(args.out_dir, datetime.datetime.now().strftime("diffgen-%Y-%m-%d-%H-%M")))
    else:
        logger.configure()

    logger.log("creating model and diffusion...")

    args.in_channels = 4
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    # batch.shape = (batch_size, 3, image_size, image_size)
    # Run the following to see what the batch looks like:
    # batch, mask, cond = next(data)
    # print(batch.shape)
    # print(mask.shape)
    # print(batch.min())
    # print(batch.max())
    # print(mask.min())
    # print(mask.max())

    # import matplotlib.pyplot as plt
    # img = batch[0].permute(1, 2, 0)
    # plt.imshow(img/255)
    # plt.savefig("img.png")
    # plt.imshow(img / 127.5 - 1)
    # plt.savefig("img1.png")
    # plt.imshow(mask[0], cmap="gray")
    # plt.savefig("mask")

    logger.log("training...")
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
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    # Remember to set --data_dir, --image_size 256, use_fp16=True
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        data_dir="",
        out_dir=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        fp16_scale_growth=1e-3,
    ))
    # Diffusion model settings
    defaults.update(dict(
        attention_resolutions="32,16,8",
        class_cond=False,
        diffusion_steps=1000,
        rescale_timesteps=True,
        learn_sigma=True,
        num_channels=256,
        noise_schedule="linear",
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=False,
        use_scale_shift_norm=True,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
