import torch
import models.archs.low_light_transformer42 as low_light_transformer42
import models.archs.discriminator_vgg_arch as SRGAN_arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'low_light_transformer42':
        netG = low_light_transformer42.low_light_transformer(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                                           groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                                           back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                                           predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                                                           w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_vgg_128_1':
        netD = SRGAN_arch.Discriminator_VGG_128_1(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_vgg_128_2':
        netD = SRGAN_arch.Discriminator_VGG_128_2(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'transformer':
        netD = SRGAN_arch.Discriminator_trans()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD
