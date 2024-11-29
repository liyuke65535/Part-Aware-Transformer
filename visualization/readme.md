# Attention Rollout

We updated the visualization codes based on https://github.com/jacobgil/vit-explain. See my examples in visualization/test.jpg.

## How to run?
Following the instruction below.
```
cd visualization

python vit_explain.py --save_path xxx --data_path xxx --vit_path xxx --pat_path xxx --pretrain_path xxx
```

For more details, please check visualization/vit_explain.py.

## No ideal reults?

You can modify the options of the line 67 in visualization/vit_explain.py.

Moreover, learn about attention fusion in visualization/vit_rollout/vit_rollout.py.
