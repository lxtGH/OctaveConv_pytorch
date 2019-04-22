# simple scripts to test convolution module
import torch


def predictImage(model):
    for i in range(100):
        img = torch.Tensor(4, 3, 256, 256).cuda()
        with torch.no_grad():
            import time
            torch.cuda.synchronize()
            start = time.time()
            out = model(img)
            torch.cuda.synchronize()
            print('Totoal Speed: {} fps.'.format(1.0 / (time.time() - start)))


if __name__ == '__main__':
    from nn.resnet import resnet50
    from nn.AdaptiveConvResnet import PixelAwareResnet50
    model = PixelAwareResnet50().cuda()
    model.eval()
    predictImage(model)
    """
    Octave Conv is half speed than Original Resnet. I guess the current implementation using nn.Conv2d.
    Using F.conv2d is a little faster than nn.Conv2d.
    single 1080-ti:
    F.conv2d:  46 fps 
    nn.conv2d: 42 fps
    """