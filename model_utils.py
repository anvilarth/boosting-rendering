import torch
import torch.nn as nn 
import torch.nn.functional as F
from models.nerf import build_nerf
from torchvision import models


def create_model(args):
    model = build_nerf(args)
    return model

class ConsistencyLoss(nn.Module):
    def __init__(self, backbone, train_imgs, device='cuda:0', lambda_=1):
        super().__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True).to(device)
        else:
            raise NotImplementedError
            
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.lambda_ = lambda_
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.real_imgs_embeddings = torch.cat([self.backbone(F.interpolate(img.permute(2, 0, 1).unsqueeze(0), (224, 224))) for img in train_imgs]).squeeze()
    
    def forward(self, indices, pred_images):
        target_phi = self.real_imgs_embeddings[indices]
        rendered_phi = self.backbone(pred_images).squeeze()
        return self.lambda_ * self.mse_loss(target_phi, rendered_phi)