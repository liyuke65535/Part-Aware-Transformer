# Part-Aware-Transformer
Official repo for "Part-Aware Transformer for Generalizable Person Re-identification" [ICCV 2023]

Here are some instructions to run our code.
Our code is based on [TransReID](https://github.com/damo-cv/TransReID), thanks for their excellent work.

## 1. Clone this repo
```
git clone https://github.com/liyuke65535/Part-Aware-Transformer.git
```

## 2. Prepare your environment
```
conda create -n screid python==3.10
conda activate screid
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
```
