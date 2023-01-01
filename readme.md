
# Bowtie Networks: Generative modeling for joint few-shot recognition and novel-view synthesis

This is the repository for [*Bowtie Networks: Generative modeling for joint few-shot recognition and novel-view synthesis*](https://arxiv.org/abs/2008.06981), published at ICLR 2021.  


## Discription

Due to the memory constrain, for the classification model, we first pretrain a feature extration network `student network` to transfer the images to feature vectors. The student network works on the `64 x 64`
resolution and is trained with knowledge distillation teacher network on full-scale images. 

After that, we store the feature vectors, together with downsampled images, to a numpy file. For the classification task, we train classifiers on the feature level.

## How to run
The model run on two stages, train (`gan.train`) on base classes and few-shot tune (`gan.few`) on novel classes. See `train_cars.sh` for a sample training.

dataset and pre-trained student network model folder: [here](https://drive.google.com/drive/folders/1bl88wMnqRt-v4dz9K8KzetQAb-e5zoI9?usp=sharing) 

## Acknowledgement
Our code is based on the awesome work of [Hologan](https://github.com/thunguyenphuoc/HoloGAN). The parameters are almost the same as them.

## Citation

```
@inproceedings{bao2021botie,
    Author = {Zhipeng Bao, Yu-Xiong Wang and Martial Hebert},
    Title = {Bowtie Networks: Generative modeling for joint few-shot recognition and novel-view synthesis},
    Booktitle = {ICLR},
    Year = {2021},
}
```
