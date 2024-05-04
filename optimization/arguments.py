import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        "-id", "--input_dir", type=str, help="The directory of the input images", required=False
    )
    parser.add_argument(
        "-td", "--target_dir", type=str, help="The directory of the target images", required=False
    )
    parser.add_argument(
        "-i", "--init_image", type=str, help="The path to the source image input", required=False
    )
    parser.add_argument(
        "-tg", "--target_image", type=str, help="The path to the target style image", required=False
    )
    parser.add_argument(
        "-ps", "--progressive_sample", help="Sample intermediate steps during reverse diffusion", action="store_true"
    )
    parser.add_argument(
        "-fti", "--find_target_image", help="Find the target image using the input image", action="store_true"
    )
    parser.add_argument(
        "-inp", "--inpainting", help="Perform inpainting using the segmentation mask", action="store_true"
    )
    parser.add_argument(
        "-sty", "--style", help="Apply styling to the image", action="store_true"
    )
    parser.add_argument(
        "-saug", "--style_aug", help="Augment by applying styles", action="store_true"
    )
    parser.add_argument(
        "-spi", "--sample_per_image", help="Sample multiple images per input", action="store_true"
    )
    parser.add_argument(
        "-cp", "--cluster_path", type=str, help="The path to the image clusters", required=False
    )
    parser.add_argument(
        "-dd", "--data_dir", type=str, help="The path to the image dataset", required=False
    )
    parser.add_argument(
        "-mp", "--model_path", type=str, help="The path to the model", required=False
    )
    parser.add_argument(
        "-cmd", "--cluster_model_dir", type=str, help="The path to the directory of cluster models", required=True
    )
    parser.add_argument(
        "-n", "--num_samples", type=int, help="The number of samples to generate", default=5
    )
    parser.add_argument(
        "-si", "--start_index", type=int, help="The cluster index to start sampling from", default=0
    )
    parser.add_argument(
        "-ei", "--end_index", type=int, help="The cluster index to end sampling", default=1
    )
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=80,
    )
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="200",
    )
    parser.add_argument(
        "--style_skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=80,
    )
    parser.add_argument(
        "--style_timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="200",
    )
    parser.add_argument(
        "--ddim",
        help="Indicator for using DDIM instead of DDPM",
        action="store_true",
    )
    
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[128, 256, 512],
    )
    parser.add_argument(
        "--clip_models",
        help="List for CLIP models",
        nargs="+",
        default=['RN50', 'RN50x4', 'ViT-B/32', 'RN50x16', 'ViT-B/16'],
    )
    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=8)
    parser.add_argument("--diff_iter", type=int, help="The number of augmentation", default=50)

    # Style losses
    parser.add_argument(
        "--lambda_trg",
        type=float,
        help="style loss for target style image",
        default=10000,
    )
    parser.add_argument(
        "--l2_trg_lambda",
        type=float,
        help="l2 loss for target style image",
        default=10000,
    )
    parser.add_argument(
        "--lambda_dir_cls", type=float, help="semantic divergence loss", default=40000,
    )
    # ----------------- #
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=200,
    )
    parser.add_argument(
        "--vit_lambda", type=float, help="total vit loss", default=1,
    )
    # Content losses
    parser.add_argument(
        "--lambda_ssim", type=float, help="key self similarity loss", default=0,
    )
    
    parser.add_argument(
        "--lambda_contra_ssim", type=float, help="contrastive loss for keys", default=0,
    )
    parser.add_argument(
        "--lambda_vgg",
        type=float,
        help="",
        default=100,#100,
    )
    parser.add_argument(
        "--lambda_zecon",
        type=float,
        help="",
        default=500,#500,
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        help="",
        default=5000, #5000,
    )
    # ----------------- #
    
    parser.add_argument(
        "--id_lambda", type=float, help="identity loss", default=100,
    )
    parser.add_argument(
        "--resample_num", type=float, help="resampling number", default=10,
    )
    parser.add_argument("--seed", type=int, help="The random seed")
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=10)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=1,
    )

    parser.add_argument(
        "--use_ffhq",
        action="store_true",
    )
    parser.add_argument(
        "--use_prog_contrast",
        action="store_true",
    )
    parser.add_argument(
        "--use_range_restart",
        action="store_true",
    )
    parser.add_argument(
        "--use_colormatch",
        action="store_true",
    )
    parser.add_argument(
        "--use_noise_aug_all",
        action="store_true",
    )
    parser.add_argument(
        "--regularize_content",
        action="store_true",
    )
    
    args = parser.parse_args()
    return args
