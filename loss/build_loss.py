import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ce_labelSmooth import CrossEntropyLabelSmooth as CE_LS

feat_dim_dict = {
    'local_attention_vit': 768,
    'part_token_vit': 768,
    'part_token_vit_hint': 768,
    'patch': 768,
    'vit': 768,
    'transreid_ssl': 768,
    'swin': 1024,
    'resnet18': 512,
    'resnet34': 512
}

def build_loss(cfg, num_classes):    # modified by gu
    name = cfg.MODEL.NAME # add by lyk
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.NAME not in feat_dim_dict.keys():
        feat_dim = 2048
    else:
        feat_dim = feat_dim_dict[cfg.MODEL.NAME]
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if name == 'local_attention_vit' and cfg.MODEL.PC_LOSS:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        else:
            xent = CE_LS(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax': # softmax loss only
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    # softmax & triplet
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet' or 'GS':
        def loss_func(score, feat, target, domains=None, t_domains=None, all_posvid=None, soft_label=False, soft_weight=0.1, soft_lambda=0.2):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if name == 'local_attention_vit' and cfg.MODEL.PC_LOSS:
                        ID_LOSS = xent(score, target, all_posvid=all_posvid, soft_label=soft_label,soft_weight=soft_weight, soft_lambda=soft_lambda)
                    else:
                        ID_LOSS = xent(score, target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                TRI_LOSS = triplet(feat, target)[0]
                # DOMAIN_LOSS = xent(domains, t_domains)
                return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                else:
                    return F.cross_entropy(score, target) + \
                            triplet(feat, target)[0] + \
                            cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                    'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


