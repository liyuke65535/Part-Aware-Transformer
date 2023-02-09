import logging
import os
import random
# from threading import local
from model.backbones.vit_pytorch import deit_tiny_patch16_224_TransReID, part_attention_deit_small, part_attention_deit_tiny, part_attention_vit_base, part_attention_vit_base_p32, part_attention_vit_large, part_attention_vit_small, vit_base_patch32_224_TransReID, vit_large_patch16_224_TransReID
import torch
import torch.nn as nn

from .backbones.resnet import BasicBlock, ResNet, Bottleneck
from .backbones import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID

# alter this to your pre-trained file name
lup_path_name = {
    'vit_base_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
    'vit_small_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
}

# alter this to your pre-trained file name
imagenet_path_name = {
    'vit_large_patch16_224_TransReID': 'jx_vit_large_p16_224-4ee7a4dc.pth',
    'vit_base_patch16_224_TransReID': 'jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_base_patch32_224_TransReID': 'jx_vit_base_patch32_224_in21k-8db57226.pth', 
    'deit_base_patch16_224_TransReID': 'deit_base_distilled_patch16_224-df68dfff.pth',
    'vit_small_patch16_224_TransReID': 'vit_small_p16_224-15ec54c9.pth',
    'deit_small_patch16_224_TransReID': 'deit_small_distilled_patch16_224-649709d9.pth',
    'deit_tiny_patch16_224_TransReID': 'deit_tiny_distilled_patch16_224-b40b3cf7.pth'
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, model_name, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        # model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 2048
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
            model_path = os.path.join(model_path_base, \
                "resnet18-f37072fd.pth")
            print('using resnet18 as a backbone')
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet34-b627a593.pth")
            print('using resnet34 as a backbone')
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet50-0676ba61.pth")
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
            model_path = os.path.join(model_path_base, \
                "resnet101-63fe2227.pth")
            print('using resnet101 as a backbone')
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            model_path = os.path.join(model_path_base, \
                "resnet152-394f9c45.pth")
            print('using resnet152 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # self.pool = nn.Linear(in_features=16*8, out_features=1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x) # B, C, h, w
        
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = self.pool(x.flatten(2)).squeeze() # is GAP harming generalization?

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))


class build_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super(build_vit, self).__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
            
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            if 'bottleneck' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

'''
part attention vit
'''
class build_part_attention_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory, pretrain_tag='imagenet'):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        if pretrain_tag == 'lup':
            path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: part token vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            pretrain_tag=pretrain_tag)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        layerwise_tokens = self.base(x) # B, N, C
        layerwise_cls_tokens = [t[:, 0] for t in layerwise_tokens] # cls token
        part_feat_list = layerwise_tokens[-1][:, 1: 4] # 3, 768

        layerwise_part_tokens = [[t[:, i] for i in range(1,4)] for t in layerwise_tokens] # 12 3 768
        feat = self.bottleneck(layerwise_cls_tokens[-1])

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, layerwise_cls_tokens, layerwise_part_tokens
        else:
            return feat if self.neck_feat == 'after' else layerwise_cls_tokens[-1]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))        

__factory_T_type = {
    'vit_large_patch16_224_TransReID': vit_large_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_base_patch32_224_TransReID': vit_base_patch32_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'deit_tiny_patch16_224_TransReID': deit_tiny_patch16_224_TransReID,
}

__factory_LAT_type = {
    'vit_large_patch16_224_TransReID': part_attention_vit_large,
    'vit_base_patch16_224_TransReID': part_attention_vit_base,
    'vit_base_patch32_224_TransReID': part_attention_vit_base_p32,
    'deit_base_patch16_224_TransReID': part_attention_vit_base,
    'vit_small_patch16_224_TransReID': part_attention_vit_small,
    'deit_small_patch16_224_TransReID': part_attention_deit_small,
    'deit_tiny_patch16_224_TransReID': part_attention_deit_tiny,
}

def make_model(cfg, modelname, num_class, sd_flag=False, head_flag=False, camera_num=None, view_num=None):
    if modelname == 'vit':
        model = build_vit(num_class, cfg, __factory_T_type)
        print('===========building vit===========')
    elif modelname == 'part_attention_vit':
        model = build_part_attention_vit(num_class, cfg, __factory_LAT_type)
        print('===========building our part attention vit===========')
    else:
        model = Backbone(modelname, num_class, cfg)
        print('===========building ResNet===========')
    ### count params
    model.compute_num_params()
    return model