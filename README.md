# Beyond Convolution
##  ~~OctaveConv_pytorch~~
## Pytorch implementation of recent operators 
  This is **third parity** implementation(un-official) of Following Paper.
  1. Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution(ICCV 2019).
  [paper](https://arxiv.org/pdf/1904.05049)
  ![](fig/octave_conv.png)
  2. Adaptively Connected Neural Networks.(CVPR 2019)
  [paper](https://arxiv.org/abs/1904.03579)
  ![](fig/adaptive_conv.png)
  3. Res2net:A New Multi-scale Backbone Architecture(PAMI 2019)
  [paper](https://arxiv.org/abs/1904.01169)
  ![](fig/res2net.png)
  4. ScaleNet:Data-Driven Neuron Allocation for Scale Aggregation Networks (CVPR2019)
  [paper](https://arxiv.org/pdf/1904.09460)
  ![](fig/sablock.png)
  5. SRM : A Style-based Recalibration Module for Convolutional Neural Networks
  [paper](https://arxiv.org/abs/1903.10829)
  ![](fig/srm.png)
  6. SEnet: Squeeze-and-Excitation Networks(CVPR 2018) [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)
  7. GEnet: Exploiting Feature Context in Convolutional Neural Networks(NIPS 2018) [paper](https://papers.nips.cc/paper/8151-gather-excite-exploiting-feature-context-in-convolutional-neural-networks.pdf)
  8. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks [paper](https://arxiv.org/abs/1910.03151)
  9. SK-Net: Selective Kernel Networks(CVPR 2019) [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Selective_Kernel_Networks_CVPR_2019_paper.pdf)
  10. More Net will be added.
  
### Plan
1. add Res2Net bolock with SE-layer (done)
2. add Adaptive-Convolution: both pixel-aware and dataset-aware (done)
3. Train code on Imagenet. (done)
4. Add SE-like models. (done)
5. Keep tracking with new proposed operators. (-)

### Usage
   check model files under the fig/nn floder.
   
```python
from lib.nn.OCtaveResnet import resnet50
from lib.nn.res2net import se_resnet50
from lib.nn.AdaptiveConvResnet import PixelAwareResnet50, DataSetAwareResnet50

model = resnet50().cuda()
model = se_resnet50().cuda()
model = PixelAwareResnet50().cuda()
model = DataSetAwareResnet50().cuda()

```
### Training

see exp floder for the detailed information

### CheckPoint


## Reference and Citation:
 
  1. OctaveConv: MXNet implementation [here](https://github.com/terrychenism/OctaveConv)
  2. AdaptiveCov: Offical tensorflow implementation [here](https://github.com/wanggrun/Adaptively-Connected-Neural-Networks)  
  3. ScaleNet: [here](https://github.com/Eli-YiLi/ScaleNet)
  4. SGENet:[here](https://github.com/implus/PytorchInsight)
  
  Please consider cite the author's paper when using the code for your research.
## License
    MIT License
