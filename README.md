# Image Enhancement Based on Pigment Representation


Se-Ho Lee, Keunsoo Ko, Seung-Wook Kim

![Example 1](./framework.png)



## Introduction
we present a novel and efficient image enhancement method based on pigment representation. Unlike conventional methods where the color transformation is restricted to pre-defined color spaces like RGB, our method dynamically adapts to input content by transforming RGB colors into a high-dimensional feature space referred to as \textit{pigments}. The proposed pigment representation offers adaptability and expressiveness, achieving superior image enhancement performance. The proposed method involves transforming input RGB colors into high-dimensional pigments, which are then reprojected individually and blended to refine and aggregate the information of the colors in pigment spaces. Those pigments are then transformed back into RGB colors to generate an enhanced output image. The transformation and reprojection parameters are derived from the visual encoder which adaptively estimates such parameters based on the content in the input image. Extensive experimental results demonstrate the superior performance of the proposed method over state-of-the-art methods in image enhancement tasks, including image retouching and tone mapping, while maintaining relatively low computational complexity and small model size.


## Train
To train a model from scratch, simply run:

```
CUDA_VISIBLE_DEVICES=0 python main.py
```
for the Adobe5K dataset.

The dataset should be placed at `../DB/Enhancement_DB/Adobe5k_480p_train_test/`

We will update this repository soon to include test scripts, pretrained model weights, and detailed instructions for reproducing the results.
