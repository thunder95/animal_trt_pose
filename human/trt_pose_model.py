from paddle.vision.models import resnet18
from paddle import nn
import paddle
import numpy as np

class ResNetBackbone(nn.Layer):
    def __init__(self, resnet):
        super(ResNetBackbone, self).__init__()
        self.resnet = resnet
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x) # /4
        x = self.resnet.layer2(x) # /8
        x = self.resnet.layer3(x) # /16
        x = self.resnet.layer4(x) # /32
        return x

class UpsampleCBR(nn.Sequential):
    def __init__(self, input_channels, output_channels, count=1, num_flat=0):
        layers = []
        for i in range(count):
            if i == 0:
                inch = input_channels
            else:
                inch = output_channels
                
            layers += [
                nn.Conv2DTranspose(inch, output_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2D(output_channels),
                nn.ReLU()
            ]
            for i in range(num_flat):
                layers += [
                    nn.Conv2D(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2D(output_channels),
                    nn.ReLU()
                ]
            
        super(UpsampleCBR, self).__init__(*layers)

class CmapPafHeadAttention(nn.Layer):
    def __init__(self, input_channels, cmap_channels, paf_channels, upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHeadAttention, self).__init__()
        self.cmap_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
        self.paf_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
        self.cmap_att = nn.Conv2D(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
        self.paf_att = nn.Conv2D(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
            
        self.cmap_conv = nn.Conv2D(upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = nn.Conv2D(upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        xc = self.cmap_up(x)
        ac =  nn.functional.sigmoid(self.cmap_att(xc))
        xp = self.paf_up(x)
        ap =  nn.functional.tanh(self.paf_att(xp))
        return self.cmap_conv(xc * ac), self.paf_conv(xp * ap)
    
#print("----===>", cmap_channels, paf_channels, upsample_channels, feature_channels, num_upsample, num_flat)
#----===> 18 42 256 512 3 0

def get_model(cmap_channels=18,  paf_channels=42):
    upsample_channels = 256
    feature_channels = 512
    num_upsample = 3
    num_flat = 0
    return  nn.Sequential(
        ResNetBackbone(resnet18(pretrained=True)),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample, num_flat)
    )

if __name__ == '__main__':
    input = paddle.ones((1, 3, 224, 224))
    model = get_model()
    model.eval()
    wgts = paddle.load("trt_pose.pdparams")
    model.set_state_dict(wgts)
    output = model(input)

    np.save("out_0", output[0].numpy())
    np.save("out_1", output[1].numpy()) # 在atol=1e-5下结果是对齐的
    # print(output)
