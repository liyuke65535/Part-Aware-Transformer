from processor.part_attention_vit_processor import part_attention_vit_do_train_with_amp
from processor.ori_vit_processor_with_amp import ori_vit_do_train_with_amp
from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.build_loss import build_loss
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import loss as Patchloss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # build DG train loader
    train_loader = build_reid_train_loader(cfg)
    # build DG validate loader
    val_name = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes = len(train_loader.dataset.pids)
    model_name = cfg.MODEL.NAME
    model = make_model(cfg, modelname=model_name, num_class=num_classes, camera_num=None, view_num=None)
    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in cfg.MODEL.NAME: # trick from moco v3
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad = False
        print("====== freeze patch_embed for stability ======")

    loss_func, center_cri = build_loss(cfg, num_classes=num_classes)

    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)
    
    ################## patch loss ####################
    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion = Patchloss.Pedal(scale=cfg.MODEL.PC_SCALE, k=cfg.MODEL.CLUSTER_K).cuda()
    if cfg.MODEL.SOFT_LABEL and cfg.MODEL.NAME == 'part_attention_vit':
        print("========using soft label========")
    ################## patch loss ####################
    
    do_train_dict = {
        'part_attention_vit': part_attention_vit_do_train_with_amp
    }
    if model_name not in do_train_dict.keys():
        ori_vit_do_train_with_amp(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query, args.local_rank,
            patch_centers = patch_centers,
            pc_criterion = pc_criterion
        )
    else :
        do_train_dict[model_name](
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query, args.local_rank,
            patch_centers = patch_centers,
            pc_criterion = pc_criterion
        )
