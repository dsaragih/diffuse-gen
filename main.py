from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments
import os


if __name__ == "__main__":
    args = get_arguments()
    # Loop through images in input directory
    """
    Script: 
    python /root/diffusion-gen/main.py \
    --cluster_path /root/diffusion-gen/clusters.pkl \
    --output_path /root/diffusion-gen/outputs \
    --model_path /root/diffusion-gen/logs/256models/model110000.pt \
    --diff_iter 100 \
    --timestep_respacing 200 \
    --skip_timesteps 80 \
    --model_output_size 256 \
    --num_samples 1 \
    --batch_size 1 \
    --use_noise_aug_all --use_colormatch \
    -fti -sty -inp

    python /root/diffusion-gen/main.py \
    --data_dir /root/diffusion-gen/guided_diffusion/segmented-images/masked-images \
    --output_path /root/diffusion-gen/outputs \
    --model_path /root/diffusion-gen/logs/256models/model200000.pt \
    --diff_iter 100 \
    --timestep_respacing 200 \
    --skip_timesteps 80 \
    --model_output_size 256 \
    --num_samples 1 \
    --batch_size 5 \
    --use_noise_aug_all --use_colormatch \
    -fti -sty -inp -spi

    python main.py \
    --cluster_path ./clusters.pkl \
    --output_path ./outputs \
    --model_path ./logs/256models/cluster_8_model110000.pt \
    --diff_iter 100 \
    --timestep_respacing 200 \
    --skip_timesteps 80 \
    --model_output_size 256 \
    --num_samples 5 \
    --batch_size 3 \
    --use_noise_aug_all --use_colormatch \
    -fti -sty -inp
    """
    if args.cluster_path:
        # expect model_path to be path/to/cluster_model/modelNNNNNN.pt
        base_model_path = args.model_path
        for i in range(args.start_index, 20):
            split = base_model_path.split("/")
            filename = f"cluster_{i}_" + split[-1]
            split[-1] = filename
            path = "/".join(split)
            # Check if file exists
            if not os.path.isfile(path):
                print(f"Cluster {i} does not exist")
                continue
            args.model_path = path
            image_editor = ImageEditor(args)
            image_editor.sample_image()
    else:
        image_editor = ImageEditor(args)
        image_editor.sample_image()
