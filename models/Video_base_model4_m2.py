import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, CharbonnierLoss2

import time

logger = logging.getLogger('base')


from PIL import Image
from dkm import DKMv3_outdoor, DKMv3_indoor
import numpy as np


class ltv_loss(nn.Module):
    def __init__(self, alpha=1.2, beta=1.5, eps=1e-4):
        super(ltv_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps=eps

    def forward(self, content, origin, weight):
        I = origin[:, 0:1, :, :] * 0.299 + origin[:, 1:2, :, :] * 0.587 + origin[:, 2:3, :, :] * 0.114
        L = torch.log(I + self.eps)
        dx = L[:, :, :-1, :-1] - L[:, :, :-1, 1:]
        dy = L[:, :, :-1, :-1] - L[:, :, 1:, :-1]
        # print(torch.mean(dx), torch.mean(dy))
        dx = self.beta / (torch.pow(torch.abs(dx), self.alpha) + self.eps)
        dy = self.beta / (torch.pow(torch.abs(dy), self.alpha) + self.eps)

        x_loss = dx * ((content[:, :, :-1, :-1] - content[:, :, :-1, 1:]) ** 2)
        y_loss = dy * ((content[:, :, :-1, :-1] - content[:, :, 1:, :-1]) ** 2)
        tvloss = torch.sum(x_loss + y_loss) / 2.0
        return tvloss * weight


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network

        self.dkm_model = DKMv3_indoor(path_to_weights='pretrained_model/DKMv3_indoor.pth', device=self.device)
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'cb2':
                self.cri_pix = CharbonnierLoss2().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)
            self.cri_ltv = ltv_loss()

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.real_H = data['GT'].to(self.device)

        self.var_L1 = data['LQs1'].to(self.device)
        self.real_H1 = data['GT1'].to(self.device)
    
    def get_mask(self, dark, light):
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)
        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max+0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        return mask
    
    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()

        out_rgb1, out_rgb2, \
            self.fake_H, fake1_H,  out_ill1_trans, out_ill2_trans = self.netG(self.var_L, self.var_L1)
        l_pix = self.l_pix_w * (self.cri_pix(self.fake_H, self.real_H)+self.cri_pix(fake1_H, self.real_H1))
        
        l_tv=self.cri_ltv.forward(out_ill1_trans, self.real_H.clone(), 0.1) * self.l_pix_w
        l_tv+=self.cri_ltv.forward(out_ill2_trans, self.real_H1.clone(), 0.1) * self.l_pix_w
        l_tv=l_tv/2
        

        batch_size=self.real_H.shape[0]
        loss_rgb=0
        for mm in range(batch_size):
            im1=self.real_H.clone().detach()[mm].cpu().permute(1, 2, 0).numpy()*255
            im2=self.real_H1.clone().detach()[mm].cpu().permute(1, 2, 0).numpy()*255
            height=im1.shape[0]
            width=im1.shape[1]
            H, W = 480, 640
            im1_path=Image.fromarray(im1.astype(np.uint8)).resize((W, H))
            im2_path=Image.fromarray(im2.astype(np.uint8)).resize((W, H))
            warp, certainty = self.dkm_model.match(im1_path, im2_path, device=self.device)
            certainty1=certainty[:, :W].unsqueeze(dim=0).unsqueeze(dim=0)
            certainty2=certainty[:, W:].unsqueeze(dim=0).unsqueeze(dim=0)

            self.dkm_model.sample(warp, certainty)
            out_rgb2_this=F.interpolate(out_rgb2[mm:mm+1], size=(H, W),mode='bilinear')
            out_rgb1_this=F.interpolate(out_rgb1[mm:mm+1], size=(H, W),mode='bilinear')
            im2_transfer_rgb = F.grid_sample(out_rgb2_this, warp[:,:W, 2:][None], mode="bilinear", align_corners=False)
            im1_transfer_rgb = F.grid_sample(out_rgb1_this, warp[:, W:, :2][None], mode="bilinear", align_corners=False)
            
            im2_transfer_rgb=F.interpolate(im2_transfer_rgb, size=(height, width),mode='bilinear')
            im1_transfer_rgb=F.interpolate(im1_transfer_rgb, size=(height, width),mode='bilinear')
            certainty1=F.interpolate(certainty1, size=(height, width),mode='bilinear')
            certainty2=F.interpolate(certainty2, size=(height, width),mode='bilinear')
            rgb_loss=(self.cri_pix(out_rgb1[mm:mm+1]*certainty1, im2_transfer_rgb*certainty1)+self.cri_pix(out_rgb2[mm:mm+1]*certainty2, im1_transfer_rgb*certainty2))/2
            loss_rgb+=rgb_loss
        loss_rgb=loss_rgb*1.0/batch_size*0.1

        l_final = l_pix+l_tv+loss_rgb
        l_final.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_tv'] = l_tv.item()
        self.log_dict['l_rgb2'] = loss_rgb.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if not (len(self.var_L.shape) == 4):
                self.var_L = self.var_L.unsqueeze(dim=0)
            if not (len(self.var_L1.shape) == 4):
                self.var_L1 = self.var_L1.unsqueeze(dim=0)
            self.out_rgb1, _, self.fake_H, _, self.out_ill1_trans, _ = self.netG(self.var_L, self.var_L1)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()

        out_dict['rlt3'] = self.out_rgb1.detach()[0].float().cpu()

        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['ill'] = self.var_L1.detach()[0].float().cpu()
        out_dict['rlt2'] = self.out_ill1_trans.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del self.real_H
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
