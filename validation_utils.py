import torch 
import torch.nn.functional as F
import numpy as np
import wandb

from models.rendering import get_rays_shapenet, sample_points, volume_render
from ssim import SSIM

def validate(model, imgs, poses, hwf, bound, num_samples, raybatch_size, scene_idx, pixel_features=None):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)
    ssim_calc = SSIM()
    view_psnrs = []
    view_ssims = []

    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, raybatch_size):
                if pixel_features is not None:
                    test_ray_features = pixel_features[i:i+raybatch_size]
                else:
                    test_ray_features = None

                rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size], test_ray_features)
                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+raybatch_size],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            error = F.mse_loss(img, synth)
            ssim_metric = ssim_calc(synth.permute(2, 0, 1).unsqueeze(0), img.permute(2, 0, 1).unsqueeze(0))
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
            view_ssims.append(ssim_metric)

    images = wandb.Image(np.concatenate((img.cpu().numpy(), synth.cpu().numpy()), axis=1), caption="Top: Output, Bottom: Input")
          
    wandb.log({"image_"+str(scene_idx): images})
    
    scene_psnr = torch.stack(view_psnrs).mean()
    scene_ssim = torch.stack(view_ssims).mean()
    return scene_psnr, scene_ssim