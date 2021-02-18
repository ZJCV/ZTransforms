# ZTransforms

## 实现背景

参考[实现背景](./background.md)

## 增强架构

图像增强操作如下：

* 几何变换(13)
    * [Resize](./resize.md)：随机缩放
    * Pad：填充
    * CenterCrop：中心裁剪
    * RandomCrop：随机裁剪
    * RandomResizedCrop：随机缩放裁剪
    * RandomHorizontalFlip：随机水平翻转
    * RandomVerticalFlip：随机竖直翻转
    * FiveCrop：5次裁剪(左上角、左下角、右上角、右下角、中心)
    * TenCrop：5次裁剪加上水平（或者垂直，自己指定）翻转后的5次裁剪
    * RandomRotation：随机旋转
    * LinearTransformation：线性转换
    * RandomAffine：随机仿射
    * RandomPerspective：随机透视
* 颜色变换(12)
    * Normalize：标准化
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
* 格式转换(4)
    * ToTensor：转换为Tensor
    * PILToTensor：PIL Image转换为Tensor
    * ConvertImageDtype：转换图像数据格式
    * ToPILImage：转换为PIL Image
* 组合操作(5)
    * Compose
    * Lambda
    * RandomApply：对增强列表中的操作按概率执行
    * RandomChoice：随机选择一个增强列表中的操作
    * RandomOrder：随机排序增强列表
  