# OctaveConv_pytorch
## Pytorch implementation of Octave convolution with other similar operation
  This is **third parity** implementation of Following Paper:
  1. Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution.[paper]((https://arxiv.org/pdf/1904.05049.pdf))
  ![](fig/octave_conv.png)
  2. Adaptively Connected Neural Networks.[paper](https://arxiv.org/abs/1904.03579)
  ![](fig/adaptive_conv.png)
  3. Res2net:[paper](https://arxiv.org/abs/1904.01169)
  ![](fig/res2net.png)
## Plan
1. Add more convolution operator (Res2Net, Adaptive-Convolution)
2. Train on Cifar/ImageNet

### Usage

```python
from nn.OCtaveResnet import resnet50

model = resnet50()
```


## Reference:
  1. OctaveConv: MXNet implementation[here](https://github.com/terrychenism/OctaveConv)
  2. AdaptiveCov: Offical tensorflow implementation[here](https://github.com/wanggrun/Adaptively-Connected-Neural-Networks)  


