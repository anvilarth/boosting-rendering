import argparse
import json
import copy
import torch
import torch.nn.functional as F
import tqdm
import os
import wandb
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from model_utils import create_model, ConsistencyLoss
from validation_utils import validate
from models.rendering import get_rays_shapenet, sample_points, volume_render
from utils.shape_video import create_360_video 
from ssim import SSIM
from torchvision import models

# os.environ['WANDB_SILENT']="true"
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='/scratch/andrey/cv/project/nerf-meta/configs/shapenet/chairs.json')
parser.add_argument('--num_train_views', type=int, default=3)
parser.add_argument('--num_steps', type=int, default=int(2e5))
parser.add_argument('--scene_idx', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--checkpoint_name', type=str, default='')
parser.add_argument('--pixel', action='store_true', default=False)
parser.add_argument('--diet', action='store_true', default=False)
parser.add_argument('--transfer', action='store_true', default=False)
parser.add_argument('--disable_logs', action='store_true', default=False)

args = parser.parse_args()

wandb.init(project="dl_project", 
            entity='neus_team', 
            name='scene=' + str(args.scene_idx) +'-n_views=' + str(args.num_train_views), 
            group='vanilla-nerf',
            mode= 'disabled' if args.disable_logs else 'online')

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Dataset loading
val_set = build_shapenet(image_set="val", dataset_root=args.dataset_root, splits_path=args.splits_path, num_views=50)

imgs, poses, hwf, bound = val_set[args.scene_idx]
imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)

train_imgs, _, test_imgs = torch.split(imgs, [args.num_train_views, 25-args.num_train_views, 25], dim=0)
train_poses, _, test_poses = torch.split(poses, [args.num_train_views, 25-args.num_train_views, 25], dim=0)

### Model creation
model = create_model(args).to(device)

optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
pixels = train_imgs.reshape(-1, 3)

if args.pixel:
    encoder = models.resnet18(pretrained=True).to(device)
    backbone = torch.nn.Sequential(*(list(encoder.children())[:-2]))
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval();

    upsampled_train_imgs = F.interpolate(train_imgs.permute(0, 3, 1, 2), size=(224, 224))
    upsampled_test_imgs = F.interpolate(test_imgs.permute(0, 3, 1, 2), size=(224, 224))

    with torch.no_grad():
        tmp1 = backbone(upsampled_train_imgs)
        tmp2 = backbone(upsampled_test_imgs)

    feature_map1 = F.interpolate(tmp1, size=(128, 128), mode='bilinear')
    feature_map2 = F.interpolate(tmp2, size=(128, 128), mode='bilinear')
    train_pixel_features = feature_map1.permute(0, 2, 3, 1).reshape(-1, 512)
    test_pixel_features = feature_map1.permute(0, 2, 3, 1).reshape(-1, 512)

else:
    train_pixel_features = None
    test_pixel_features = None

rays_o, rays_d = get_rays_shapenet(hwf, train_poses)
full_ray_origins, full_ray_directions = get_rays_shapenet(hwf, train_poses)
rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

num_rays = rays_d.shape[0]
consistency_criterion = ConsistencyLoss('resnet50', train_imgs)
K = 100

for step in tqdm.tqdm(range(args.num_steps)):
    indices = torch.randint(num_rays, size=[args.train_batchsize])
    raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
    pixelbatch = pixels[indices] 

    if args.pixel:
        train_ray_features = train_pixel_features[indices]
    else:
        train_ray_features = None


    t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                args.num_samples, perturb=True)
    optim.zero_grad()

    rgbs, sigmas = model(xyz, train_ray_features)
    colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
    loss = F.mse_loss(colors, pixelbatch)

    
    if args.diet:
        if step % K == 0:
            image_index = np.random.randint(len(train_imgs))
            target_image, loss_ray_origins, loss_ray_directions = train_imgs[image_index], full_ray_origins[image_index], full_ray_directions[image_index]
            loss_ray_origins, loss_ray_directions = loss_ray_origins.reshape(-1, 3), loss_ray_directions.reshape(-1, 3)
        
            t_vals, xyz = sample_points(loss_ray_origins, loss_ray_directions, bound[0], bound[1],
                                        args.num_samples, perturb=False)

            synth = []
            num_rays = loss_ray_directions.shape[0]
            for i in (range(0, num_rays, args.test_batchsize)):

                if args.pixel:
                    train_ray_features = train_pixel_features[i:i+args.test_batchsize]
                else:
                    train_ray_features = None


                rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize], train_ray_features)

                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+args.test_batchsize],
                                            white_bkgd=True)
                synth.append(color_batch)
                
            synth = torch.cat(synth, dim=0)
            synth = synth.reshape_as(target_image)
            synth = synth.permute(2, 0 ,1)
            synth = torch.unsqueeze(synth, 0)

            synth = F.interpolate(synth, (224, 224))

            consistency_loss = consistency_criterion(image_index, synth)
            loss += consistency_loss
        
    loss.backward()
    optim.step()
    
    if step % 20000 == 0:
        psnr, ssim = validate(model, test_imgs, test_poses, hwf, bound, args.num_samples, args.test_batchsize, scene_idx=args.scene_idx, pixel_features=test_pixel_features)
        wandb.log({'psnr': psnr, 'ssim': ssim})
        if args.checkpoint_name != '':
            torch.save({'state_dict':model.state_dict(), 'optimizer_dict':optimizer.state_dict()}, args.checkpoint_name + 'scene=' + str(args.scene_idx) +'-n_views=' + str(args.num_train_views) + '-step=' + str(step) +  '.ckpt')

psnr, ssim = validate(model, test_imgs, test_poses, hwf, bound, args.num_samples, args.test_batchsize, scene_idx=args.scene_idx, pixel_features=test_pixel_features)

wandb.log({'final_psnr': psnr, 'final_ssim': ssim})
if args.checkpoint_name != '':
    torch.save({'state_dict':model.state_dict(), 'optimizer_dict':optimizer.state_dict()}, args.checkpoint_name + 'scene=' + str(args.scene_idx) +'-n_views=' + str(args.num_train_views) + '-step=' + str(step) +  '.ckpt')

video_name = 'scene=' + str(args.scene_idx) +'-n_views=' + str(args.num_train_views)

savedir = Path('./video')
savedir.mkdir(exist_ok=True)

create_360_video(args, model, hwf, bound, device, video_name, savedir)