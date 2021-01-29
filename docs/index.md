# ZTransforms

`github`地址：[ZJCV/ZTransforms](https://github.com/ZJCV/ZTransforms)

## 实现背景

[PyTorch](https://github.com/pytorch/pytorch)提供了官方的数据增强实现：[transforms](https://github.com/pytorch/vision/tree/master/torchvision/transforms)。该模块基于`PIL`为后端进行数据增强操作，其优缺点如下：

* 优点：
    1. 简洁清晰的数据架构
    2. 简单易懂的数据处理流程
    3. 完善的文档介绍
* 缺点：
    1. 基于`PIL`后端，使用`RGB PIL Image`图像格式；
    2. 提供的图像增强功能有限；且相较于其他库的执行速度慢

针对于执行速度问题，`torchvision`也意识到了这一点，从`0.8.0`开始进行了改进

```
Prior to v0.8.0, transforms in torchvision have traditionally been PIL-centric and presented multiple limitations due to that. Now, since v0.8.0, transforms implementations are Tensor and PIL compatible and we can achieve the following new features:

transform multi-band torch tensor images (with more than 3-4 channels)
torchscript transforms together with your model for deployment
support for GPU acceleration
batched transformation such as for videos
read and decode data directly as torch tensor with torchscript support (for PNG and JPEG image formats)
```

* 一方面通过[Pillow-SIMD](https://github.com/uploadcare/pillow-simd)提高`PIL`的执行速度；
* 另一方面通过`Tensor`操作来实现`GPU`加速

在网上找到两个数据增强库：

* [imgaug](https://github.com/aleju/imgaug)：其实现了更多的数据增强操作；
* [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)：其在不同的后端（`pytorch/imgaug/opencv`）中找出各自最快的增强函数（参考[Benchmarking
  results](https://github.com/albumentations-team/albumentations#benchmarking-results)）

上述两个数据增强库均实现了类似于`transforms`的数据流操作方式。不过相对而言，个人还是最喜欢官方的实现和使用方式

新建这个代码库，基于[transforms](https://github.com/pytorch/vision/tree/master/torchvision/transforms)，在原有功能中添加`albumentation/imgaug`后端实现，同时添加新的数据增强操作

## 架构

模式如下：

* <函数名>（后端实现一/后端实现二/...）：<输入格式一/输入格式二/...>

图像增强操作如下：

* 几何变换
  * Resize(pil/pytorch/albumentation): PIL Image/tensor/ndarray
  * CenterCrop(pil/pytorch/albumentation): PIL Image/tensor/ndarray
  * Pad(pil/pytorch/albumentation): PIL Image/tensor/ndarray
  * RandomCrop(pil/pytorch/albumentation): PIL Image/tensor/ndarray
* 颜色变换
* 格式转换
  * ToTensor(pil/pytorch): ndarray/PIL Image/tensor
  * PILToTensor(pil/pytorch/): PIL Image
  * ConvertImageDtype(pil/pytorch): tensor
  * ToPILImage(pil/pytorch): ndarray/tensor
  * Normalize(pil/pytorch): tensor
* 组合操作
  * Compose
  * Lambda
  * RandomApply
  * RandomChoice
  * RandomOrder
  