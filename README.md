# Part-Aware-Transformer
Official repo for "Part-Aware Transformer for Generalizable Person Re-identification" [ICCV 2023]

<div align=center><img src="https://github.com/liyuke65535/Part-Aware-Transformer/assets/39180877/a92d863d-43c7-48ca-b4d5-d34eef237fd5"></div>




## Abstract
Domain generalization person re-identification (DG-ReID) aims to train a model on source domains and generalize well on unseen domains.
Vision Transformer usually yields better generalization ability than common CNN networks under distribution shifts. 
However, Transformer-based ReID models inevitably over-fit to domain-specific biases due to the supervised learning strategy on the source domain.
We observe that while the global images of different IDs should have different features, their similar local parts (e.g., black backpack) are not bounded by this constraint. 
Motivated by this, we propose a pure Transformer model (termed Part-aware Transformer) for DG-ReID by designing a proxy task, named Cross-ID Similarity Learning (CSL), to mine local visual information shared by different IDs. This proxy task allows the model to learn generic features because it only cares about the visual similarity of the parts regardless of the ID labels, thus alleviating the side effect of domain-specific biases. 
Based on the local similarity obtained in CSL, a Part-guided Self-Distillation (PSD) is proposed to further improve the generalization of global features. 
Our method achieves state-of-the-art performance under most DG ReID settings. 

## Framework
<div align=center><img src="https://github.com/liyuke65535/Part-Aware-Transformer/assets/39180877/f400b553-5a58-4238-9cde-a0d66e232586"></div>

## Visualizations
<div align=center><img src="https://github.com/liyuke65535/Part-Aware-Transformer/assets/39180877/a0f002c3-ef46-4d63-a3f0-e90dfe0ed61c"></div>
<div align=center><img src="https://github.com/liyuke65535/Part-Aware-Transformer/assets/39180877/191e0958-46b1-4262-b850-e3264e919a4d"></div>

# Instructions

Here are some instructions to run our code.
Our code is based on [TransReID](https://github.com/damo-cv/TransReID), thanks for their excellent work.

## 1. Clone this repo
```
git clone https://github.com/liyuke65535/Part-Aware-Transformer.git
```

## 2. Prepare your environment
```
conda create -n pat python==3.10
conda activate pat
bash enviroments.sh
```

## 3. Prepare pretrained model (ViT-B) and datasets
You can download it from huggingface, rwightman, or else where.
For example, pretrained model is avaliable at [ViT-B](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).

As for datasets, follow the instructions in [MetaBIN](https://github.com/bismex/MetaBIN#8-datasets).

## 4. Modify the config file
```
# modify the model path and dataset paths of the config file
vim ./config/PAT.yml
```

## 5. Train a model
```
bash run.sh
```

## 6. Evaluation only
```
# modify the trained path in config
vim ./config/PAT.yml

# evaluation
python test.py --config ./config/PAT.yml
```
## Citation
If you find this repo useful for your research, you're welcome to cite our paper.
```
@inproceedings{ni2023part,
  title={Part-Aware Transformer for Generalizable Person Re-identification},
  author={Ni, Hao and Li, Yuke and Gao, Lianli and Shen, Heng Tao and Song, Jingkuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11280--11289},
  year={2023}
}
```
