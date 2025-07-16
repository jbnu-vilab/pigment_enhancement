# Image Enhancement Based on Pigment Representation


Se-Ho Lee, Keunsoo Ko, Seung-Wook Kim

![Example 1](./framework.png)



## Introduction
We present a novel and efficient image enhancement method based on pigment representation. Unlike conventional methods where the color transformation is restricted to pre-defined color spaces like RGB, our method dynamically adapts to input content by transforming RGB colors into a high-dimensional feature space referred to as pigments. The proposed pigment representation offers adaptability and expressiveness, achieving superior image enhancement performance. The proposed method involves transforming input RGB colors into high-dimensional pigments, which are then reprojected individually and blended to refine and aggregate the information of the colors in pigment spaces. Those pigments are then transformed back into RGB colors to generate an enhanced output image. The transformation and reprojection parameters are derived from the visual encoder which adaptively estimates such parameters based on the content in the input image. Extensive experimental results demonstrate the superior performance of the proposed method over state-of-the-art methods in image enhancement tasks, including image retouching and tone mapping, while maintaining relatively low computational complexity and small model size.

## Environment

The experiments were conducted using the following software environment:

- **PyTorch**: 1.12
- **Torchvision**: 0.13.0  
- **CUDA**: 11.3  
- **Python**: 3.8
- **OS**: Ubuntu 20.04 LTS

## Dataset

The following datasets were used in this project:

- **PPR10K**: [Download Link](https://github.com/csjliang/PPR10K)
- **Adobe5K**: [Download Link](https://data.csail.mit.edu/graphics/fivek/)

After downloading, please place the datasets in the following directories:

- `PPR10K` dataset should be located at:
```  
../data/train_val_images_tif_360p/
                              ├── train/
                              │ ├── input/
                              │ ├── target_A/
                              │ ├── target_B/
                              │ └── target_C/
                              └── test/
                              │ ├── input/
                              │ ├── target_A/
                              │ ├── target_B/
                              │ └── target_C/
```
- `Adobe5K` dataset should be located at:
```
../data/Adobe5k_480p/
                ├── test/
                │ ├── input/
                │ └── user-c/
                └── train/
                │ ├── input/
                └──── user-c/
```
We conduct experiments using a **480p downsampled version** of the original 4K Adobe5K images.  

The downsampling is applied to the **short side** of each image to maintain aspect ratio.  

We split the dataset into **training** and **testing** sets manually, as shown in the directory structure above.

Make sure that the directory structures match the expected format for training and evaluation scripts.

## Pretrained Models

You can download the pretrained models from the following links:

- **Adobe5K**: [Dropbox](https://...)  
- **PPR10K**: [Google Drive](https://...)  
- Checkpoints include `.pth` files, which should be placed in the following directory:

```
./model/
```

## Train
To train a model from scratch, simply run:

```
python main.py --dataset adobe5k --model_name adobe5k_res5 --backbone_type 5
```
Please note that the Adobe5K dataset should be placed at `../data/adobe5k/`

You can specify the backbone type using the `--backbone_type` argument:
- `--backbone_type 5` for the 5-layer backbone [25]
- `--backbone_type 18` for ResNet-18  
- `--backbone_type 34` for ResNet-34  

These are described in detail in the paper.


## Test
To evaluate a trained model, simply run:

```
python main.py --dataset adobe5k --model_name adobe5k_res5 --backbone_type 5 --resume 2 --test 1
```

- `--resume 1` loads the checkpoint from the latest epoch.  
- `--resume 2` loads the checkpoint with the best PSNR performance.

To save output images during testing, set the following flag:

```
--saveimg 1
```
