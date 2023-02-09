import random
import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment
from PIL import Image, ImageFilter, ImageOps

from .transforms import LGT

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def build_transforms(cfg, is_train=True, is_fake=False):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # augmix augmentation
        do_augmix = cfg.INPUT.DO_AUGMIX

        # auto augmentation
        do_autoaug = cfg.INPUT.DO_AUTOAUG
        # total_iter = cfg.SOLVER.MAX_ITER
        total_iter = cfg.SOLVER.MAX_EPOCHS

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE

        # Local Grayscale Transfomation
        do_lgt = cfg.INPUT.LGT.DO_LGT
        lgt_prob = cfg.INPUT.LGT.PROB

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_mean = cfg.INPUT.REA.MEAN
        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        if do_autoaug:
            res.append(AutoAugment(total_iter))
        res.append(T.Resize(size_train, interpolation=3))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_lgt:
            res.append(LGT(lgt_prob))
        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_augmix:
            res.append(AugMix())
        # if do_rea:
        #     res.append(RandomErasing(probability=rea_prob, mean=rea_mean, sh=1/3))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
        if is_fake:
            if cfg.META.DATA.SYNTH_FLAG == 'jitter':
                res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=1.0))
            elif cfg.META.DATA.SYNTH_FLAG == 'augmix':
                res.append(AugMix())
            elif cfg.META.DATA.SYNTH_FLAG == 'both':
                res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
                res.append(AugMix())
        res.extend([
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        if do_rea:
            from timm.data.random_erasing import RandomErasing as RE
            res.append(RE(probability=rea_prob, mode='pixel', max_count=1, device='cpu'))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test, interpolation=3))
        res.extend([
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    return T.Compose(res)
