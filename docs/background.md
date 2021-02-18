
# 实现背景

[PyTorch](https://github.com/pytorch/pytorch)提供了官方数据增强实现：[transforms](https://github.com/pytorch/vision/tree/master/torchvision/transforms)。该模块基于`PIL`进行数据增强操作，其优缺点如下：

* 优点：
  1.  简洁清晰的数据架构
  2.  简单易懂的数据处理流
  3. 完善的文档介绍
* 缺点：
  1.  基于`PIL`后端，提供的图像增强功能有限
  2.  基于`PIL`后端，相较于其他库的执行速度慢
 
针对于执行速度问题，`torchvision`也意识到了这一点，从`0.8.0`开始进行了改进
  
```
Prior to v0.8.0, transforms in torchvision have traditionally been PIL-centric and presented multiple limitations due to that. Now, since v0.8.0, transforms implementations are Tensor and PIL compatible and we can achieve the following new features:

transform multi-band torch tensor images (with more than 3-4 channels)
torchscript transforms together with your model for deployment
support for GPU acceleration
batched transformation such as for videos
read and decode data directly as torch tensor with torchscript support (for PNG and JPEG image formats)
```

* 一方面通过新的后端[Pillow-SIMD](https://github.com/uploadcare/pillow-simd)来提高`PIL`的执行速度；
* 另一方面添加`PyTorch`后端来实现`GPU`加速

在网上找到两个数据增强库：

* [imgaug](https://github.com/aleju/imgaug)：其实现了更多的数据增强操作；
* [albumentations](https://github.com/albumentations-team/albumentations/tree/f2462be3a4d01c872474d0e7fc0f32f387b06340)：其在不同的后端（`pytorch/imgaug/opencv`）中找出各自最快的增强函数（参考[Benchmarking results](https://github.com/albumentations-team/albumentations#benchmarking-results)）

上述两个数据增强库均实现了类似于`transforms`的数据流操作方式。不过相对而言，个人还是最喜欢官方的实现和使用方式，所以新建这个代码库，基于[transforms](https://github.com/pytorch/vision/tree/master/torchvision/transforms)，在原有功能中添加`albumentation`后端实现，同时添加新的数据增强操作（*如果`albumentation`未实现，就使用`imgaug`实现*）
