'''
     Bilateral Segmentation Network for Real-time Semantic Segmentation
Author: Zhengwei Li
Data:  2018/12/04
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
def dconv_bn_act(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )
def conv_bn_act(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )
def bn_act(inp):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        nn.PReLU(inp)
    )
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(make_dense, self).__init__()
        
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(growthRate)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x_ = self.bn(self.conv(x))
        out = self.act(x_)
        out = torch.cat((x, out), 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, reset_channel=False):
        super(DenseBlock, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out

# ResidualDenseBlock
class ResidualDenseBlock(nn.Module):
    def __init__(self, nIn, s=4, add=True):

        super(ResidualDenseBlock, self).__init__()

        n = int(nIn//s) 

        self.conv =  nn.Conv2d(nIn, n, 1, stride=1, padding=0, bias=False)
        self.dense_block = DenseBlock(n, nDenselayer=(s-1), growthRate=n)

        self.bn = nn.BatchNorm2d(nIn)
        self.act = nn.PReLU(nIn)

        self.add = add

    def forward(self, input):

        # reduce
        inter = self.conv(input)
        combine =self.dense_block(inter)

        # if residual version
        if self.add:
            combine = input + combine

        output = self.act(self.bn(combine))
        return output
        
def conv_bn_relu(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )
def bn_relu(inp):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        nn.ReLU()
    )
# Attention Refinement Module (ARM)
class ARM(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(ARM, self).__init__()

        self.global_pool = nn.AvgPool2d(kernel_size, stride=kernel_size)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        # self.bn = nn.BatchNorm1d(in_channels)
        self.sigmod = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear')

    def forward(self, input):
        x = self.global_pool(input)

        x = self.conv_1x1(x)
        # x = self.bn(x)

        x = self.sigmod(x)
        x = self.up(x)

        out = torch.mul(input, x)
        return out

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, in_channels, classes, kernel_size):
        super(FFM, self).__init__()
        self.conv_bn_relu = conv_bn_relu(in_channels, classes)

        self.global_pool = nn.AvgPool2d(kernel_size, stride=kernel_size)
        self.conv_1x1_1 = nn.Conv2d(classes, classes, 1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv_1x1_2 = nn.Conv2d(classes, classes, 1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear')


    def forward(self, sp, cx):
        input = torch.cat((sp, cx), 1)
        feather = self.conv_bn_relu(input)

        x = self.global_pool(feather)

        x = self.conv_1x1_1(x)
        x = self.relu(x)
        x = self.conv_1x1_2(x)
        x = self.sigmod(x)
        x = self.up(x)

        out = torch.mul(feather, x)
        out = feather + out

        return out

class BiSeNet(nn.Module):

    def __init__(self, classes=2):

        super(BiSeNet, self).__init__()


        # -----------------------------------------------------------------
        # Spatial Path 
        # ---------------------
        self.conv_bn_relu_1 = conv_bn_relu(3, 12, kernel_size=3, stride=2, padding=1)
        self.conv_bn_relu_2 = conv_bn_relu(12, 16, kernel_size=3, stride=2, padding=1)
        self.conv_bn_relu_3 = conv_bn_relu(16, 32, kernel_size=3, stride=2, padding=1)
        # -----------------------------------------------------------------
        # Context Path 
        # ---------------------
        # input cascade
        self.cascade = nn.AvgPool2d(3, stride=2, padding=1)

        # 1/2
        self.head_conv = conv_bn_relu(3, 12, kernel_size=3, stride=2, padding=1)
        self.stage_0 = ResidualDenseBlock(12, s=3, add=True)

        # 1/4
        self.ba_1 = bn_relu(12+3)
        self.down_1 = conv_bn_relu(12+3, 24, kernel_size=3, stride=2, padding=1)
        self.stage_1 = ResidualDenseBlock(24, s=3, add=True)
        # 1/8
        self.ba_2 = bn_relu(48+3)
        self.down_2 = conv_bn_relu(48+3, 48, kernel_size=3, stride=2, padding=1)
        self.stage_2 = ResidualDenseBlock(48, s=3, add=True)
        # 1/16
        self.ba_3 = bn_relu(96+3)
        self.down_3 = conv_bn_relu(96+3, 96, kernel_size=3, stride=2, padding=1)
        self.stage_3 = nn.Sequential(ResidualDenseBlock(96, s=6, add=True),
                                     ResidualDenseBlock(96, s=6, add=True))
        # 1/32
        self.ba_4 = bn_relu(192+3)
        self.down_4 = conv_bn_relu(192+3, 192, kernel_size=3, stride=2, padding=1)
        self.stage_4 = nn.Sequential(ResidualDenseBlock(192, s=6, add=True),
                                     ResidualDenseBlock(192, s=6, add=True)) 

        self.global_pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')


        # ARM
        self.arm_s3 = ARM(96, kernel_size=16)
        self.arm_s4 = ARM(192, kernel_size=8)

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')

        # FFM
        self.ffm = FFM(320, classes, kernel_size=32)

        self.up = nn.Upsample(scale_factor=8, mode='bilinear')

        self.last_conv = nn.Conv2d(classes, classes, 1, stride=1, padding=0)

    def forward(self, input):

        # -----------------------------------------------
        # Spatial Path 
        # ---------------------
        sp = self.conv_bn_relu_1(input)
        sp = self.conv_bn_relu_2(sp)
        sp = self.conv_bn_relu_3(sp)
        # -----------------------------------------------
        # Context Path 
        # ---------------------
        input_cascade1 = self.cascade(input)
        input_cascade2 = self.cascade(input_cascade1)
        input_cascade3 = self.cascade(input_cascade2)
        input_cascade4 = self.cascade(input_cascade3)

        x = self.head_conv(input)
        # 1/2
        s0 = self.stage_0(x)

        s1_0 = self.down_1(self.ba_1(torch.cat((input_cascade1, s0),1)))
        s1 = self.stage_1(s1_0)

        s2_0 = self.down_2(self.ba_2(torch.cat((input_cascade2, s1_0, s1),1)))
        s2 = self.stage_2(s2_0)

        s3_0 = self.down_3(self.ba_3(torch.cat((input_cascade3, s2_0, s2),1)))
        s3 = self.stage_3(s3_0)

        s4_0 = self.down_4(self.ba_4(torch.cat((input_cascade4, s3_0, s3),1)))
        s4 = self.stage_4(s4_0)

        tail = self.global_pool(s4)
        tail = self.up_8(tail)

        s4_ = self.arm_s4(s4)
        s4_ = torch.mul(s4_, tail)
        s4_ = self.up_4(s4_)

        s3_ = self.arm_s3(s3)
        s3_ = self.up_2(s3_)

        cx = torch.cat((s4_, s3_), 1)
        # -----------------------------------------------
        # Fusion
        # ---------------------
        heatmap = self.ffm(sp, cx)

        heatmap = self.up(heatmap)
        out = self.last_conv(heatmap)

        return out
if __name__ == '__main__':

    model = BiSeNet(3, 'resnet101')

    x = torch.rand(2, 3, 256, 256)

    y = model(x)
