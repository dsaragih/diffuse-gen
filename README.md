# Using diffusion models to generate synthetic labeled data for medical image segmentation

Repository for the paper [Using diffusion models to generate synthetic labeled data for medical image segmentation](https://doi.org/10.1007/s11548-024-03213-z).

## Abstract
### Purpose
Medical image analysis has become a prominent area where machine learning has been applied. However, high-quality, publicly available data are limited either due to patient privacy laws or the time and cost required for experts to annotate images. In this retrospective study, we designed and evaluated a pipeline to generate synthetic labeled polyp images for augmenting medical image segmentation models with the aim of reducing this data scarcity.

### Methods
We trained diffusion models on the HyperKvasir dataset, comprising 1000 images of polyps in the human GI tract from 2008 to 2016. Qualitative expert review, Fréchet Inception Distance (FID), and Multi-Scale Structural Similarity (MS-SSIM) were tested for evaluation. Additionally, various segmentation models were trained with the generated data and evaluated using Dice score (DS) and Intersection over Union (IoU).

### Results
Our pipeline produced images more akin to real polyp images based on FID scores. Segmentation model performance also showed improvements over GAN methods when trained entirely, or partially, with synthetic data, despite requiring less compute for training. Moreover, the improvement persists when tested on different datasets, showcasing the transferability of the generated images.

### Conclusions
The proposed pipeline produced realistic image and mask pairs which could reduce the need for manual data annotation when performing a machine learning task. We support this use case by showing that the methods proposed in this study enhanced segmentation model performance, as measured by Dice and IoU scores, when trained fully or partially on synthetic data.

## Running the code

### Environment
Pytorch 1.12.1, Python 3.9

```
$ conda create --name diffgen python=3.9
$ conda activate diffgen
$ pip install ftfy regex matplotlib lpips kornia opencv-python color-matcher blobfile scikit-learn pytorch_msssim
$ pip install torch==1.12.1 torchvision==0.13.1
$ pip install git+https://github.com/openai/CLIP.git
$ (conda or pip) install mpi4py 
```

### Model download
To generate images, please download the pre-trained diffusion model(s)

Full Model [LINK](https://drive.google.com/drive/folders/1GMQiG7qQiS2hIMBAN3I_-F7gigvo9IwL?usp=drive_link) ~2 GB

Cluster (20) Models [LINK](https://drive.google.com/drive/folders/1ChgZZltlj5KlxSRg2e3hTdcsJDGjq4m4?usp=drive_link) ~40GB

download the model into ```./checkpoints``` folder.

### Clusters
The .pkl file containing the cluster assignments may be found in the ```clustering``` folder. Note that cluster assignments for 10, 20, 30, and 40 are provided, as well as a simple notebook to generate the clusters.

Unfortunately, we aren't able to provide pre-trained models for $k \neq 20$, but the training code is provided below.

### Training Models
```
# Cluster Models

python ~/diffusion-gen/guided_diffusion/BOI_train.py \
    --data_dir ~/diffusion-gen/guided_diffusion/segmented-images \
    --cluster_file ~/diffusion-gen/clusters.pkl \
    --cluster_index 1 \
    --image_size 256 \
    --out_dir ~/diffusion-gen/checkpoints \
    --batch_size 1

# Full Model

python ~/diffusion-gen/guided_diffusion/image_train.py \
    --data_dir ~/diffusion-gen/guided_diffusion/segmented-images \
    --image_size 256 \
    --out_dir ~/diffusion-gen/checkpoints \
    --batch_size 1
```
```--cluster_index``` may be changed to any cluster index from the cluster file

### Sample images from cluster models


```
python ~/diffusion-gen/main.py \
    --cluster_path ~/diffusion-gen/clusters.pkl \
    --output_path ~/diffusion-gen/cluster_image_samples \
    --model_path ~/diffusion-gen/checkpoints/cluster_models/model110000.pt \
    --diff_iter 100 \
    --timestep_respacing 200 \
    --skip_timesteps 80 \
    --model_output_size 256 \
    --num_samples 1 \
    --batch_size 1 \
    --use_noise_aug_all \
    --use_colormatch \
    -fti -sty -inp -spi
```

The cluster path, output path, and model path can be changed to your own path.

Note the format of the model path: each model will be of the form ```cluster_{cluster index}_model_{model iteration}.pt```. For example, ```cluster_0_model_110000.pt``` is the model for cluster 0 at iteration 110000. In the ```main.py``` file, we loop through all the cluster models with the given iteration number and generate samples for each cluster.

```-fti``` selects a random image from the cluster to use as the target image for image-guided generation

```-sty``` applies styling

```-inp``` uses the inpainting technique

```-spi``` instead of just sampling {num_samples} images, we sample {num_samples} images from each image

### Sample images from full model

```
python ~/diffusion-gen/main.py \
    --data_dir ~/diffusion-gen/guided_diffusion/segmented-images/masked-images \
    --output_path ~/diffusion-gen/image_samples \
    --model_path ~/diffusion-gen/logs/256models/model200000.pt \
    --diff_iter 100 \
    --timestep_respacing 200 \
    --skip_timesteps 80 \
    --model_output_size 256 \
    --num_samples 1 \
    --batch_size 1 \
    --use_noise_aug_all \
    --use_colormatch \
    -fti -sty -inp -spi
```

Refer to above section for clarification on the arguments.

Our source code rely on Blended-diffusion, guided-diffusion, flexit, splicing vit, and DiffuseIT

## Citation

If you find this work useful, please consider citing:

```
@article{saragihUsingDiffusionModels2024,
  title = {Using Diffusion Models to Generate Synthetic Labeled Data for Medical Image Segmentation},
  author = {Saragih, Daniel G. and Hibi, Atsuhiro and Tyrrell, Pascal N.},
  year = {2024},
  month = aug,
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  volume = {19},
  number = {8},
  pages = {1615--1625},
  issn = {1861-6429},
  doi = {10.1007/s11548-024-03213-z},
  urldate = {2024-08-19},
}
```