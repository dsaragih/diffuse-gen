"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os
import sys

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # Second param is shape of noise tensor x_T
        samples = sample_fn(
            model,
            (args.batch_size, 4, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            skip_timesteps=80,
            progress=True,
            device=dist_util.dev(),
            final_only=args.final_only,
        )
        logger.log(len(samples))
        samples = th.stack([sample["sample"] for sample in samples]).squeeze(1)
        logger.log(samples.shape)
        # [-1, 1] -> [0, 255]
        def preprocess(sample):
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            return sample
        sample = preprocess(samples)
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    logger.log(arr.shape)
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
    # Output array shape is (num_samples, 256, 256, 4), where the 4th channel is the mask
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=1,
        use_ddim=False,
        model_path="/home/daniel/diff-seg/core/logs/cluster_models/cluster_0_model040000.pt",
        attention_resolutions="16,8",
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
        final_only=True,
        in_channels=4, # must be 4 for segmentation
        timestep_respacing="200", # efficiency of sampling
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
