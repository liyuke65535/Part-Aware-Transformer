import argparse
import random
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from config_vis import cfg

from vit_rollout.vit_rollout import VITAttentionRollout
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import make_model

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    cfg.MODEL.PRETRAIN_PATH = args.pretrain_path

    # load part_attention_vit
    model_ours = make_model(cfg, 'part_attention_vit', num_class=1)
    model_ours.load_param(args.pat_path)
    model_ours.eval()
    model_ours.to('cuda')

    # load vanilla vit
    model_vit = make_model(cfg, 'vit', num_class=1)
    model_vit.load_param(args.vit_path)
    model_vit.eval()
    model_vit.to('cuda')

    transform = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_tensor = []

    # Prepare the original person photos
    base_dir = args.data_path
    img_path = os.listdir(base_dir)
    random.shuffle(img_path)

    length = min(30, len(img_path)) # how many photos to visualize
    img_list = []
    for pth in img_path[:length]:
        img = Image.open(base_dir+pth)
        img = img.resize((128,256))
        np_img = np.array(img)[:, :, ::-1] # BGR -> RGB
        input_tensor = transform(img).unsqueeze(0)
        input_tensor = input_tensor.cuda()
        img_list.append(np_img)

        local_flag = False

        # attention rollout
        for model in [model_ours]:
            attention_rollout = VITAttentionRollout(model, head_fusion='mean', discard_ratio=0.5) # modify head_fusion type and discard_ratio for better outputs
            masks = attention_rollout(input_tensor)

            if isinstance(masks, list):
                for msk in masks:
                    msk = cv2.resize(msk, (np_img.shape[1], np_img.shape[0]))
                    img_list.append(show_mask_on_image(np_img, msk))
                    local_flag = True
            else:
                masks = cv2.resize(masks, (np_img.shape[1], np_img.shape[0]))
                out_img = show_mask_on_image(np_img, masks)
                img_list.append(out_img)


    final_img = []
    line_len = 5 if local_flag else 3

    # concate output images in a column
    for i in range(0, len(img_list)-1, line_len):
        if i==0:
            img_line = [img_list[l] for l in range(line_len)]
            final_img = np.concatenate(img_line,axis=1)
        else:
            img_line = [img_list[i+l] for l in range(line_len)]
            x = np.concatenate(img_line,axis=1)
            final_img = np.concatenate([final_img,x],axis=0)
    
    cv2.imwrite(args.save_path, final_img)
    for i, pth in enumerate(img_path[:30]):
        print(i+1, pth)
    print(f"save to {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="path to save your attention visualized photo. E.g., /home/me/out.jpg")
    parser.add_argument("--data_path", type=str, help="path to your dataset. E.g., dataset/market1501/query")
    parser.add_argument("--pretrain_path", type=str, help="path to your pretrained vit from imagenet or else. E.g., /home/me/cpt/")
    parser.add_argument("--vit_path", type=str, help="path to your trained vanilla vit. E.g., cpt/vit.pth")
    parser.add_argument("--pat_path", type=str, help="path to your trained PAT. E.g., cpt/pat.pth")
    args = parser.parse_args()
    main(args)