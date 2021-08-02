## Introduction

TF_SPLERGE is an implementation for [SPLERGE](https://ieeexplore.ieee.org/document/8977975) which is an method for tabel structure decomposition, published on ICDAR 2019 and got the best paper in table strucuture analysis (**the paper has been put in the rootdir of the rep**). 

My blogs about this implementation is [here](https://blog.csdn.net/abandon_first/category_11000568.html). Another implementation based on Torch is [here](https://github.com/CharlesWu123/SPLERGE), which provided reference for me at the beginning.

## Requirement

tensorflow==2.2.0

opencv3==3.1.0

numpy==1.19.1

networkx==2.5.1

fitz==0.0.1

## Inference

The models I provide are **over-fitting** due to the poor **quantity** of the data . So please run the `inference.py` just for  a test.

## Train

### About The Dataset

If you just want to learn SPLERGE, you can only use the public dataset from ICDAR which **I have put into this rep**. While if you want to use this method for real work, you should prepare **your own large mount dataset** just like the paper's author. 

Code about processing the public ICDAR 2013 dataset  is in the `process_data folder`. You can process it from the beginning by running the `proc_ICDAR2013.py`. After this, you can view the label for text bounding box under `check_img` folder.

![](/home/yx-lxd/PycharmProjects/TF_SPLERGE/check_img/eu-dataset-eu-001-0.png)

Codes to generate tfrds for train is in the `gen_ground_truth` folder. You can generate the data by running `gen_gt.py`.

### About The Training

All hyperparameters about training can be changed in `split_config.py` and `merge_config.py` under the `config` folder.

You must **train for split stage** and get a model for split which can work well **firstly**. 

```
python split_train.py
```

After this, you can train the merge stage **based on the split model**. I froze the split model to train the merge model until the last. Theoretically, after training merge model for a while, you can release the split model and train the two stage together.

```
python merge_train.py
```

### About The Testing

Run the `test.py` in the rootdir.



