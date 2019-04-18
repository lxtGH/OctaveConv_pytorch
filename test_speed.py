import torch

def predictImage(model):
    for i in range(100):
        img = torch.Tensor(4, 3, 224, 224).cuda()
        with torch.no_grad():
            import time
            torch.cuda.synchronize()
            start = time.time()
            out = model(img)
            torch.cuda.synchronize()
            print('Totoal Speed: {} fps.'.format(1.0 / (time.time() - start)))



if __name__ == '__main__':
    from nn.resnet import resnet50
    from nn.OCtaveResnet import resnet50 as OCresnet50
    from nn.AdaptiveConvResnet import DataSetAwareResnet50
    model = DataSetAwareResnet50().cuda()
    # model = resnet50().cuda()
    model.eval()
    predictImage(model)