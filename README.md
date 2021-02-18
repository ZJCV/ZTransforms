<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="README.zh-CN.md">🇨🇳</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/ZTransforms.git"><img align="center" src="./imgs/ZTransforms.png"></a></div>

<p align="center">
  «ZTransforms» is an image data enhancement code base
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

based on [pytorch/vision](https://github.com/pytorch/vision/) architecture，add [albumentations](https://github.com/albumentations-team/albumentations/tree/f2462be3a4d01c872474d0e7fc0f32f387b06340) as the backend

* input image format：`numpy ndarray`
* data type：`uint8`
* channel arrangement order：`rgb`

critical dependencies's version:

* `pytorch/vision:  c1f85d34761d86db21b6b9323102390834267c9b`
* `albumentations-team/albumentations: v0.5.2`

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

[PyTorch](https://github.com/pytorch/pytorch) provides an official data enhancement implementation：[transforms](https://github.com/pytorch/vision/tree/master/torchvision/transforms)。The module performs data enhancement operation based on PIL, and its advantages and disadvantages are as follows:

* Advantages:
  1.  Simple and clear data architecture
  2.  Simple and understandable data processing flow
  3. Perfect documentation introduction
* Disadvantages:
  1.  Based on the PIL backend, the provided image enhancement function is limited
  2.  Compared with other implementations, the execution speed is not fast
 
`torchvision` is also aware of this and has made improvements since "0.8.0"

```
Prior to v0.8.0, transforms in torchvision have traditionally been PIL-centric and presented multiple limitations due to that. Now, since v0.8.0, transforms implementations are Tensor and PIL compatible and we can achieve the following new features:

transform multi-band torch tensor images (with more than 3-4 channels)
torchscript transforms together with your model for deployment
support for GPU acceleration
batched transformation such as for videos
read and decode data directly as torch tensor with torchscript support (for PNG and JPEG image formats)
```

* On the one hand, the new backend [Pill-SIMD](https://github.com/uploadcare/Pill-SIMD) is used to improve the execution speed of PIL;
* On the other hand, PyTorch backend is added to realize GPU acceleration

Two data enhancement libraries are found on the Internet, which provide detection/segmentation data enhancement in addition to classification data enhancement:

* [imgaug](https://github.com/aleju/imgaug)：Which realizes more data enhancement operations；
* [albumentations](https://github.com/albumentations-team/albumentations/tree/f2462be3a4d01c872474d0e7fc0f32f387b06340)：It finds out the fastest enhancement function in different backend (`pytorch/imgaug/opencv`) (refer to [benchmarking results](https://github.com/Albumentations-team/Albumentations#benchmarking-results))

The above two data enhancement libraries have realized the data flow operation mode similar to `transforms`。However, relatively speaking, I still like the official implementation and usage. Therefore, this code base is newly built, based on [transforms](https://github.com/pytorch/vision/tree/master/torchvision/transforms), the `albumentation` backend implementation is added to the original functions, and new data enhancement operations are also added (if `albumentation` is not implemented, use `imgaug/opencv/...` to implement it).

## Install

```
$ pip install ztransforms
```

## Usage

```
# import torchvision.transforms as transforms
import ztransforms.cls as transforms
...
...
```

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [pytorch/vision](https://github.com/pytorch/vision)
* [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations/tree/f2462be3a4d01c872474d0e7fc0f32f387b06340)
* [aleju/imgaug](https://github.com/aleju/imgaug)
* [opencv/opencv](https://github.com/opencv/opencv)

```
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}

@misc{imgaug,
  author = {Jung, Alexander B.
            and Wada, Kentaro
            and Crall, Jon
            and Tanaka, Satoshi
            and Graving, Jake
            and Reinders, Christoph
            and Yadav, Sarthak
            and Banerjee, Joy
            and Vecsei, Gábor
            and Kraft, Adam
            and Rui, Zheng
            and Borovec, Jirka
            and Vallentin, Christian
            and Zhydenko, Semen
            and Pfeiffer, Kilian
            and Cook, Ben
            and Fernández, Ismael
            and De Rainville, François-Michel
            and Weng, Chi-Hung
            and Ayala-Acevedo, Abner
            and Meudec, Raphael
            and Laporte, Matias
            and others},
  title = {{imgaug}},
  howpublished = {\url{https://github.com/aleju/imgaug}},
  year = {2020},
  note = {Online; accessed 01-Feb-2020}
}
```

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/ZJCV/ZTransforms/issues) or submit PRs.

Small note:

* Git submission specifications should be complied with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the[standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## License

[Apache License 2.0](LICENSE) © 2021 zjykzj