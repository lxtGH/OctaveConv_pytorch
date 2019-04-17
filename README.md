# OctaveConv_pytorch
## Pytorch implementation of Octave convolution with other similar operation
  This is **third parity** implementation(un-official) of Following Paper which are talked in[Recente_Convolution.pdf](https://github.com/lxtGH/OctaveConv_pytorch/blob/master/Recent_Convolution.pdf):
  1. Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution.
  [paper](https://arxiv.org/pdf/1904.05049.pdf)
  ![](fig/octave_conv.png)
  2. Adaptively Connected Neural Networks.(CVPR 2019)
  [paper](https://arxiv.org/abs/1904.03579)
  ![](fig/adaptive_conv.png)
  3. Res2net:A New Multi-scale Backbone Architecture
  [paper](https://arxiv.org/abs/1904.01169)
  ![](fig/res2net.png)
   
  
## Plan
1. add Res2Net bolock with SE-layer (done)
2. add Adaptive-Convolution: both pixel-aware and dataset-aware (done)
3. add HetConv(optional): if I have time :)  
3. Train on Cifar () 
4. Train on Imagenet (): Who can help me train this repo on Imagenet
### Usage
   check model files under the nn floder.
   
```python
from nn.OCtaveResnet import resnet50
from nn.res2net import se_resnet50
from nn.AdaptiveConvResnet import PixelAwareResnet50, DataSetAwareResnet50

model = resnet50().cuda()
model = se_resnet50().cuda()
model = PixelAwareResnet50().cuda()
model = DataSetAwareResnet50().cuda()

```


## Reference:
  1. OctaveConv: MXNet implementation[here](https://github.com/terrychenism/OctaveConv)
  2. AdaptiveCov: Offical tensorflow implementation[here](https://github.com/wanggrun/Adaptively-Connected-Neural-Networks)  

## License
    MIT License
