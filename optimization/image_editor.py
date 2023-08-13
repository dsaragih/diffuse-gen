import os
import blobfile as bf
import pickle
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils_visualize.metrics_accumulator import MetricsAccumulator
from utils_visualize.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import cv2
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from torchvision import models
from optimization.losses import range_loss, d_clip_loss, get_features, zecon_loss_direct
# import lpips
import numpy as np
from src.vqc_core import *
from model_vit.loss_vit import Loss_vit
from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils_visualize.visualization import show_tensor_image, show_editied_masked_image
from pathlib import Path
from id_loss import IDLoss
import datetime
from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer

class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        self.model_path = self.args.model_path
        # os.makedirs(self.args.output_path, exist_ok=True)
        base_dir = datetime.datetime.now().strftime("diffgen-%Y-%m-%d-%H-%M")
        if self.args.cluster_path:
            cluster_index = int(self.model_path.split("/")[-1].split("_")[1])
            base_dir = f"cluster{cluster_index}-{base_dir}"

        self.ranked_results_path = Path(self.args.output_path, base_dir)
        self.root_dir = Path(__file__).parent.parent.as_posix()

        os.makedirs(self.ranked_results_path, exist_ok=True)
        
        logger.configure(dir=str(self.ranked_results_path))
        logger.log("Root dir:", self.root_dir)
        logger.log("Args:", self.args)
        logger.log("Loading model...")
        logger.log("Model path:", self.model_path)

        if self.args.seed is not None:
            logger.log("Setting seed:", self.args.seed)
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "16,8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
                "in_channels": 4 # must be 4 for segmentation
            }
        )

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        logger.log("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model_config.update(
            {
                "timestep_respacing": self.args.style_timestep_respacing
            }
        )
        _, self.style_diffusion = create_model_and_diffusion(**self.model_config)

        def prep_model(model):
            model.load_state_dict(
                torch.load(
                self.model_path,
                map_location="cpu",
                )
            )
            model.requires_grad_(False).eval().to(self.device)
            for name, param in model.named_parameters():
                if "qkv" in name or "norm" in name or "proj" in name:
                    param.requires_grad_()
            if self.model_config["use_fp16"]:
                model.convert_to_fp16()
            return model
        self.model = prep_model(self.model)
        with open(f"{self.root_dir}/model_vit/config.yaml", "r") as ff:
            config = yaml.safe_load(ff)

        cfg = config
        """
        lambda_ssim = l_ssim
        lambda_contra_ssim = l_cont
        lambda_dir_cls = l_sem
        lambda_trg = l_sty

        Want to replace lambda_ssim, lambda_contra_ssim with zecon loss
        """
        self.VIT_LOSS = Loss_vit(cfg, lambda_ssim=self.args.lambda_ssim,lambda_dir_cls=self.args.lambda_dir_cls,lambda_contra_ssim=self.args.lambda_contra_ssim,lambda_trg=args.lambda_trg).eval()
        
        self.cm = ColorMatcher()

        # self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

        if self.args.lambda_vgg > 0:
            self.vgg = models.vgg19(pretrained=True).features
            self.vgg.to(self.device)
            self.vgg.eval().requires_grad_(False)
        
        self.vgg_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def noisy_aug(self,t,x,x_hat):
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        x_mix = x_hat * fac + x * (1 - fac)
        return x_mix
    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep
    
    def zecon_loss(self, x_in, y_in, t):
        loss = zecon_loss_direct(self.model, x_in, y_in, torch.zeros_like(t,device=self.device))
        return loss.mean()
    
    def vgg_loss(self,x_in, y_in):
        content_features = get_features(self.vgg_normalize(x_in), self.vgg)
        target_features = get_features(self.vgg_normalize(y_in), self.vgg)
        loss = 0

        loss += torch.mean((target_features['conv1_1'] - content_features['conv1_1']) ** 2)
        loss += torch.mean((target_features['conv2_1'] - content_features['conv2_1']) ** 2)
        # loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        # loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        return loss.mean()
    
    def cnt_mse_loss(self, x_in, y_in):
        loss = mse_loss(x_in, y_in)
        return loss.mean()

    def _list_image_files_recursively(self, data_dir):
        results = []
        for entry in sorted(bf.listdir(data_dir)):
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                results.append(full_path)
            elif bf.isdir(full_path):
                results.extend(self._list_image_files_recursively(full_path))
        return results
    
    def _load_cluster(self, cluster_path):
        # Model file name is cluster_i_modelN.pt
        cluster_index = int(self.model_path.split("/")[-1].split("_")[1])
        logger.log("Loading cluster", cluster_index, "from", cluster_path)
        with open(cluster_path, "rb") as f:
            cluster = pickle.load(f)
        return cluster[cluster_index]
    
    def _get_target_image_and_mask(self, img_paths, it=None, exclude_path=None):
        if exclude_path is not None:
            img_paths = [p for p in img_paths if p != exclude_path]
        len_img_paths = len(img_paths)
        rand_file = img_paths[it % len_img_paths] if it is not None else random.choice(img_paths)
        # rand_file of the form .../guided_diffusion/data
        # Replace . with self.root_dir
        rand_file = bf.join(self.root_dir, rand_file[2:]) if self.args.cluster_path else rand_file
        mask_path = bf.join(bf.dirname(bf.dirname(rand_file)), 'masks', bf.basename(rand_file))

        self.target_image_pil = Image.open(rand_file).convert("RGB")
        self.target_mask_pil = Image.open(mask_path).convert("L")
        self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)
        self.target_mask_pil = self.target_mask_pil.resize(self.image_size, Image.LANCZOS)

        # Dilate mask
        if self.args.inpainting:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask_arr = np.array(self.target_mask_pil)
            mask_arr = np.where(mask_arr > 0, 1, 0).astype(np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
            self.target_mask_pil = Image.fromarray(mask_arr.astype(np.float64))

        self.target_image = (
            TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        self.target_mask = (
            TF.to_tensor(self.target_mask_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        self.target_image = self.target_image.repeat(self.args.batch_size, 1, 1, 1)
        self.target_mask = self.target_mask.repeat(self.args.batch_size, 1, 1, 1)

        return rand_file, mask_path
    def _get_init_image_and_mask(self, img_paths, it=None, exclude_path=None):
        if exclude_path is not None:
            img_paths = [p for p in img_paths if p != exclude_path]
        len_img_paths = len(img_paths)
        rand_file = img_paths[it % len_img_paths] if it is not None else random.choice(img_paths)
        rand_file = bf.join(self.root_dir, rand_file[2:]) if self.args.cluster_path else rand_file
        mask_path = bf.join(bf.dirname(bf.dirname(rand_file)), 'masks', bf.basename(rand_file))

        self.init_image_pil = Image.open(rand_file).convert("RGB")
        self.init_mask_pil = Image.open(mask_path).convert("L")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)
        self.init_mask_pil = self.init_mask_pil.resize(self.image_size, Image.LANCZOS)

        # Dilate mask
        if self.args.inpainting:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask_arr = np.array(self.init_mask_pil)
            mask_arr = np.where(mask_arr > 0, 1, 0).astype(np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
            self.target_mask_pil = Image.fromarray(mask_arr.astype(np.float64))

        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        # Tile to batch size
        self.init_image = self.init_image.repeat(self.args.batch_size, 1, 1, 1)
        self.init_mask = (
            TF.to_tensor(self.init_mask_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        self.init_mask = self.init_mask.repeat(self.args.batch_size, 1, 1, 1)

        return rand_file, mask_path

    def _save(self, all_images, styled_images):
        # Apply np.array to all values in dict
        img_dict = {k: np.array(v) for k, v in all_images.items()}
        arr = next(iter(img_dict.values()))
        img_dict_shape = list(arr.shape)
        img_dict_shape[0] = img_dict_shape[0] * len(img_dict)
        # arr = arr[: args.num_samples]
        logger.log(f"Shape of img_dict: {img_dict_shape}")
        styled_dict = {k: np.array(v) for k, v in styled_images.items()}
        if styled_dict == {}: # If no styling
            styled_dict = img_dict
        styled_arr = next(iter(styled_dict.values()))
        styled_dict_shape = list(styled_arr.shape)
        styled_dict_shape[0] = styled_dict_shape[0] * len(styled_dict)
        logger.log(f"Shape of styled_dict: {styled_dict_shape}")
        # arr = np.array(all_images)
        # styled_arr = np.array(styled_images)

        shape_str = "x".join([str(x) for x in img_dict_shape])
        out_path = os.path.join(self.ranked_results_path, f"samples_{shape_str}.pkl")
        styled_shape_str = "x".join([str(x) for x in styled_dict_shape])
        styled_out_path = os.path.join(self.ranked_results_path, f"styled_samples_{styled_shape_str}.pkl")

        logger.log(f"saving to {out_path}")
        # np.savez(out_path, arr)
        # np.savez(styled_out_path, styled_arr)
        # Save dicts as pkl
        with open(out_path, 'wb') as f:
            pickle.dump(img_dict, f)
        with open(styled_out_path, 'wb') as f:
            pickle.dump(styled_dict, f)

    def sample_image(self):
        shape = (self.args.batch_size, 4, self.model_config["image_size"], self.model_config["image_size"])
        self.init_image = torch.zeros(shape).to(self.device)
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        it = 0
        if self.args.init_image is not None:
            self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
            self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.init_image = (
                TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1))
        
        self.target_image = None
        if self.args.target_image is not None:
            self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
            self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.target_image = (
                TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
            )
        elif self.args.find_target_image and (self.args.cluster_path or self.args.data_dir): # find target image
            all_files = self._load_cluster(self.args.cluster_path) if self.args.cluster_path else self._list_image_files_recursively(self.args.data_dir)
            
            style_img_path = None

            if False and self.args.init_image is not None:
                best_loss = self.VIT_LOSS.calculate_global_ssim_loss(self.init_image, self.target_image) + self.VIT_LOSS.calculate_contra_ssim_loss(self.init_image, self.target_image)
                best_tgt_img_pil = self.target_image_pil
                
                for path in all_files:
                    target_image_pil = Image.open(path).convert("RGB")
                    target_image_pil = target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
                    target_image = (
                        TF.to_tensor(target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
                    )
                    ssim_loss = self.VIT_LOSS.calculate_global_ssim_loss(self.init_image, target_image)
                    cont_ssim_loss = self.VIT_LOSS.calculate_contra_ssim_loss(self.init_image, target_image)
                    # Save target_image with lowest ssim + contrastive ssim
                    if ssim_loss + cont_ssim_loss < best_loss:
                        best_loss = ssim_loss + cont_ssim_loss
                        best_tgt_img_pil = target_image_pil
                self.target_image_pil = best_tgt_img_pil
                self.target_image = (
                    TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
                )
       
        self.prev = self.init_image.detach()
        self.flag_resample=False
        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        def cond_fn(x, t, y=None):
            # if self.args.prompt == "":
            #     return torch.zeros_like(x)
            self.flag_resample=False
            with torch.enable_grad():
                frac_cont=1.0
   
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                loss = torch.tensor(0) # default value
                if self.init_image.eq(0).all(): # reconstruction
                    noise = torch.randn_like(x)
                    nonzero_mask = (
                        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                    )  # no noise when t == 0
                    x_in = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

                    if self.target_image is not None:
                        loss = loss + mse_loss(x_in[:, :3, ...], self.target_image) * self.args.l2_trg_lambda
                else: # styling
                    if self.args.use_noise_aug_all:
                        x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                    else:
                        x_in = out["pred_xstart"]
                    # self.init_image = (B,4,H,W)
                    x_in3 = x_in[:, :3, ...]
                    # init_image_batch = torch.tile(self.init_image[:3, ...], dims=(self.args.batch_size, 1, 1, 1))
                    # zecon_init_image_batch = torch.tile(self.init_image, dims=(self.args.batch_size, 1, 1, 1))
                    # self.prev = torch.tile(self.prev[:3, ...], dims=(self.args.batch_size, 1, 1, 1))
                    init_image_batch = self.init_image[:, :3, ...]
                    zecon_init_image_batch = self.init_image
                    self.prev = self.prev[:, :3, ...]

                    if self.args.vit_lambda != 0:     
                        # self.init_image is x_src  
                        if t[0].item()>self.args.diff_iter : # directional cls
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in3, init_image_batch,self.prev,use_dir=True,frac_cont=frac_cont,target = self.target_image)
                        else:
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in3,init_image_batch,self.prev,use_dir=False,frac_cont=frac_cont,target = self.target_image)
                        loss = loss + vit_loss

                    if self.args.range_lambda != 0:
                        r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                        loss = loss + r_loss
                        self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                    if self.target_image is not None:
                        loss = loss + mse_loss(x_in3, self.target_image) * self.args.l2_trg_lambda

                    self.prev = x_in3.detach().clone()

                    # ------------------  New Losses ------------------
                    fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]             

                    if not self.args.use_noise_aug_all:
                        x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                        
                    if self.args.lambda_zecon != 0:
                        y_t = self.diffusion.q_sample(zecon_init_image_batch,t)
                        y_in = zecon_init_image_batch * fac + y_t * (1 - fac)
    
                        zecon_loss = self.zecon_loss(x_in, y_in,t) * self.args.lambda_zecon
                        loss = loss + zecon_loss
                        self.metrics_accumulator.update_metric("zecon_loss", zecon_loss.item())
                    
                    if self.args.lambda_vgg != 0 and t[0].item() < 800:
                        y_t = self.diffusion.q_sample(init_image_batch,t)
                        y_in = init_image_batch * fac + y_t * (1 - fac)

                        vgg_loss = self.vgg_loss(x_in3, y_in) * self.args.lambda_vgg
                        loss = loss + vgg_loss
                        self.metrics_accumulator.update_metric("vgg_loss", vgg_loss.item())
                    if self.args.lambda_mse != 0 and t[0].item() < 700:
                        y_t = self.diffusion.q_sample(init_image_batch, t)
                        y_in = init_image_batch * fac + y_t * (1 - fac)

                        cnt_mse_loss = self.cnt_mse_loss(x_in3, y_in) * self.args.lambda_mse
                        loss = loss + cnt_mse_loss
                        self.metrics_accumulator.update_metric("cnt_mse_loss", cnt_mse_loss.item())

                    # ------------------  New Losses End ------------------
                    
                    if self.args.use_range_restart:
                        if t[0].item() < total_steps:
                            if r_loss>0.01:
                                    self.flag_resample =True
            return (-torch.autograd.grad(loss, x)[0] if not loss.eq(0).all() else loss), self.flag_resample

        # [-1, 1] -> [0, 255]
        def preprocess(sample):
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            return sample
        
        def refine_mask(sample_batch):
            mask = sample_batch[:, 3, ...]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
            mask = mask.cpu().numpy()
            for i in range(mask.shape[0]):
                mask[i] = cv2.morphologyEx(mask[i], cv2.MORPH_OPEN, kernel)
            mask = torch.from_numpy(mask).to(sample_batch.device)
            sample_batch[:, 3, ...] = mask
            return sample_batch

        
        # all_images = []
        # styled_images = []
        all_images = {}
        styled_images = {}
        total_style_steps = self.style_diffusion.num_timesteps - self.args.style_skip_timesteps
        save_image_interval = total_style_steps // 5
        num_samples = self.args.num_samples * len(all_files) if self.args.sample_per_image else self.args.num_samples

        logger.log(f"Sampling {num_samples * self.args.batch_size} images")
        while it < num_samples:
            style_img_path, _ = self._get_target_image_and_mask(all_files, it=it)
            it += 1
            logger.log(f"Style image {style_img_path}")

            if not self.args.style_aug:
                sample_func = (
                    self.diffusion.ddim_sample_loop if self.args.ddim else self.diffusion.p_sample_loop
                )

                samples = sample_func(
                    self.model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={},
                    cond_fn=None,
                    progress=True,
                    skip_timesteps=self.args.skip_timesteps,
                    init_image=torch.cat([self.target_image, self.target_mask], dim=1) if self.args.inpainting else None,
                    postprocess_fn=None,
                    device=self.device,
                    final_only=(not self.args.progressive_sample),
                    inpainting=self.args.inpainting
                )
                if self.flag_resample:
                    continue
                samples = torch.stack([refine_mask(sample["sample"]) for sample in samples]).squeeze(0)
                
                logger.log(samples.shape) # [batch_size, 4, 256, 256]
            # NOTE: If we are styling, we want to predict x_0, so return sample["pred_xstart"], otherwise, we sample, so return sample["sample"]
            else:
                src_image_path, src_mask = self._get_init_image_and_mask(all_files, style_img_path)
                logger.log(f"Source image {src_image_path} with mask {src_mask}")
                samples = torch.cat([self.init_image, self.init_mask], dim=1)
                # Tile to batch size
                samples = samples.repeat(self.args.batch_size, 1, 1, 1)

            if self.args.style or self.args.style_aug:
                logger.log("Styling samples...")
                self.init_image = samples
                self.prev = self.init_image.detach()
                style_func = (
                    self.style_diffusion.ddim_sample_loop_progressive
                    if self.args.ddim
                    else self.style_diffusion.p_sample_loop_progressive
                )
                styled_samples = style_func(
                    self.model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=self.args.style_skip_timesteps,
                    init_image=self.init_image,
                    postprocess_fn=None,
                    randomize_class=True,
                    inpainting=False
                )
                for j, styled_sample in enumerate(styled_samples):
                    should_save_image = j % save_image_interval == 0 or j == total_style_steps - 1
                    if should_save_image:
                        styled_img = styled_sample["pred_xstart"]
                        styled_samples = preprocess(styled_img)
                        styled_image = styled_samples.cpu().numpy() 
                        # Last in batch, shape (W, H, 4)
                if self.args.use_colormatch and self.init_image is not None:
                    for img in styled_image:
                        src_image = Normalizer(img[..., :3]).type_norm()
                        arr_pil = np.asarray(self.target_image_pil)
   
                        trg_image = Normalizer(arr_pil).type_norm()
                        img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                        img_res = Normalizer(img_res).uint8_norm()
                        img = np.concatenate([img_res, img[..., -1:]], axis=-1)
                        # logger.log("Styled image shape colormatch", img.shape) # W, H, 4
                        # styled_images.extend([styled_image])
                        curr = styled_images.get(style_img_path, [])
                        curr.extend([img])
                        styled_images[style_img_path] = curr
                    logger.log("Styled image shape colormatch", styled_images[style_img_path][-1].shape) # W, H, 4
                else:
                    curr = styled_images.get(style_img_path, [])
                    curr.extend([img for img in styled_image])
                    styled_images[style_img_path] = curr   

            samples = preprocess(samples)
            # all_images.extend([sample.cpu().numpy() for sample in samples])
            # Add to dict
            # Empty list if key doesn't exist
            curr = all_images.get(style_img_path, [])
            curr.extend([sample.cpu().numpy() for sample in samples])
            # image, mask = curr[-1][..., :3], curr[-1][..., -1:]
            # import matplotlib.pyplot as plt
            # plt.imshow(np.array(image))
            # plt.savefig(f"image_{it}.png")
            # plt.imshow(np.array(mask), cmap="gray")
            # plt.savefig(f"mask_{it}.png")
            all_images[style_img_path] = curr

            if it % 50 == 0:
                logger.log(f"Saving {it} images...")
                self._save(all_images, styled_images)

        if it % 50 != 0: # prevent saving twice
            logger.log(f"Saving {it} images...")        
            self._save(all_images, styled_images)