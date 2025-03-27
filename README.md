# Classification of Thoracic Diseases on Chest X-ray Images

## Introduction

*Medical imaging* is an indispensable technology for modern medicine, and *deep learning*, DL, has been applied to this field since early times. A typical example is the modeling of reading and diagnosis of medical images such as X-ray, CT, and MRI. If we can construct a model to estimate the classification and location of diseases in medical images, we can expect to reduce the burden on the image reading physicians, equalize diagnostic criteria, and predict early diagnosis and disease onset.

In the early days of DL for medical images, it was difficult to prepare a widely shared dataset such as the `ImageNet` dataset for general object recognition due to patient consent, privacy protection, etc. However, gradually datasets of sufficient size for DL training have been publicly available. One example is the dataset of chest X-ray images released in 2017 by the *National Institutes of Health*, NIH[^Wang].

Here, we introduce a model for classifying thoracic diseases using this chest X-ray images dataset. In addition, we discuss how to reduce the computational time required for training, by using *distributed data parallel*, DDP.

## Dataset

We use the `ChestX-ray14` dataset for training. This dataset is constructed from the `ChestX-ray8` dataset, which was built from over 30,000 chest X-ray images published in 2017, and later increased to 14 annotated diseases.

The `ChestX-ray14` dataset consists of 112,120 chest radiographs from 30,805 patients, with multiple disease labels corresponding to each image from 14 different diseases. The labels of the dataset are based on the findings extracted using *natural language processing*, NLP, from *electronic health records*, EHR.

We split the dataset into a training set of 70%, a validation set of 10%, and a test set of 20% for training and validation. A breakdown of the diseases in the dataset and their percentages is shown in Table 1.

|                   | **train (%)** | **val (%)** | **test (%)** |
|:----------------- | ------------: | ----------: | -----------: |
|Atelectasis        |       10.2    |      10.0   |       10.8   |
|Cardiomegaly       |        2.5    |       2.1   |        2.6   |
|Effusion           |       11.8    |      11.5   |       12.3   |
|Infiltration       |       17.7    |      18.0   |       17.6   |
|Mass               |        5.1    |       5.6   |        5.1   |
|Nodule             |        5.6    |       5.5   |        6.0   |
|Pneumonia          |        1.2    |       1.2   |        1.1   |
|Pneumothorax       |        4.7    |       4.5   |        4.9   |
|Consolidation      |        4.2    |       4.0   |        4.3   |
|Edema              |        2.2    |       1.8   |        1.8   |
|Emphysema          |        2.3    |       1.9   |        2.3   |
|Fibrosis           |        1.5    |       1.5   |        1.6   |
|Pleural Thickening |        2.9    |       3.3   |        3.3   |
|Hernia             |        0.2    |       0.4   |        0.2   |

**Table 1. Percentage of thoracic diseases in each split of the `ChestX-ray14` dataset for training, validation, and test.**

## CheXNet

For thoracic disease classification, we use `CheXNet` proposed in 2017 by *P. Rajpurkar* et al[^Rajpurkar]. `CheXNet` is a model based on `DenseNet-121`, a typical *convolutional neural networks*, CNN, which uses chest X-ray images as input to perform multi-label classification for thoracic diseases. The difference from the original `DenseNet-121` is that an output layer is added to classify 14 diseases. This implementation uses an improved version of the original `CheXNet` model with a sigmoid function added to the final layer.

For training, we load the pretrained weights of `DenseNet-121` on the `ImageNet` dataset and then perform fine-tuning on the `ChestX-ray14` dataset.

## Distributed Data Parallel

This implementation can use *graphics processing units*, GPU, or `Habana Gaudi` as *accelerator*. If multiple accelerators are specified, training is performed by *distributed data parallel*, DDP. For example, the following command performs DDP training with 8 `Habana Gaudi` accelerators.

```console
$ torchrun --nnodes=1 --nproc_per_node=8 main.py --hpu
```

The results of training using `Habana Gaudi` showed that it took about 860 seconds to train about 15,000 images. As a comparison, we measured the training time with `Tesla V100`, which took about 1,045 seconds for the same training. There was no significant difference in the training loss.

## Acknowledgements

The use of `ChestX-ray14` dataset is advised by *N. Sato*, Kyoto University.

[^Wang]: X. Wang et al., *ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, **CVPR**, 2017.
[^Rajpurkar]: P. Rajpurkar et al, *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, **arXiv**, 2017.
