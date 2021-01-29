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

* <函数名>（后端实现一/后端实现二/...）

图像增强操作如下：

* 几何变换
  * Resize(pil/pytorch/albumentation)：随机缩放
  * Pad(pil/pytorch/albumentation)：填充
  * CenterCrop(pil/pytorch/albumentation)：中心裁剪
  * RandomCrop(pil/pytorch/albumentation)：随机裁剪
  * RandomResizedCrop(pil/pytorch/albumentation)：随机缩放裁剪
  * RandomHorizontalFlip(pil/pytorch/albumentation)：随机水平翻转
  * RandomVerticalFlip(pil/pytorch/albumentation)：随机竖直翻转
  * FiveCrop(pil/pytorch/albumentation)：5次裁剪(左上角、左下角、右上角、右下角、中心)
  * TenCrop(pil/pytorch/albumentation)：5次裁剪加上水平（或者垂直，自己指定）翻转后的5次裁剪
  * RandomRotation：随机旋转
  * LinearTransformation：线性转换
  * RandomAffine：随机仿射
  * RandomPerspective：随机透视
* 颜色变换
  * Normalize(pil/pytorch)：标准化
  * ColorJitter：颜色抖动（亮度、对比度、饱和度、色度）
  * Grayscale：灰度化
  * RandomGrayscale：随机灰度化
  * RandomErasing：随机擦除
  * GaussianBlur：高斯模糊
  * RandomInvert：随机反向像素值
  * RandomPosterize：
  * RandomSolarize
  * RandomAdjustSharpness：随机调整锐度
  * RandomAutocontrast：随机调整对比度
  * RandomEqualize：随机均衡化
* 格式转换
  * ToTensor(pil/pytorch)：转换为Tensor
  * PILToTensor(pil/pytorch/)：PIL Image转换为Tensor
  * ConvertImageDtype(pil/pytorch)：转换图像数据格式
  * ToPILImage(pil/pytorch)：转换为PIL Image
* 组合操作
  * Compose：组合
  * Lambda
  * RandomApply：对增强列表中的操作按概率执行
  * RandomChoice：随机选择一个增强列表中的操作
  * RandomOrder：随机排序增强列表
  