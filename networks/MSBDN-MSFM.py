import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_networks import Encoder_MDCBlock1, Decoder_MDCBlock1
def make_model(args, parent=False):
    return Net()
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class SPRLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SPRLayer, self).__init__()

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

        self.fc1 = nn.Conv2d(channels * 5, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out1 = self.avg_pool1(x).view(x.size(0), -1, 1, 1)
        out2 = self.avg_pool2(x).view(x.size(0), -1, 1, 1)
        out = torch.cat((out1, out2), 1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
class SPRModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SPRModule, self).__init__()

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

        self.fc1 = nn.Conv2d(channels * 5, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out1 = self.avg_pool1(x).view(x.size(0), -1, 1, 1)
        out2 = self.avg_pool2(x).view(x.size(0), -1, 1, 1)
        out = torch.cat((out1, out2), 1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
class MSAModule(nn.Module):
    def __init__(self, inplanes, scale=3, stride=1, stype='normal'):
        """ 
        MSAModule 实现了论文中的 MSPA 三个主要组件：
        1. HPC (Hierarchical Ghost Convolution) 多尺度特征提取
        2. SPR (Spatial Pyramid Recalibration) 空间金字塔重校准
        3. Softmax 跨尺度权重归一化
        """
        super(MSAModule, self).__init__()
        self.width = inplanes
        self.nums = scale
        self.stride = stride
        assert stype in ['stage', 'normal'], 'One of these is supported (stage or normal)'
        self.stype = stype

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        ### ===== ① HPC 模块开始：Split → Conv → Concat =====
        for i in range(self.nums):
            if self.stype == 'stage' and self.stride != 1:
                self.convs.append(convdilated(self.width, self.width, stride=stride, dilation=int(i + 1)))
            else:
                self.convs.append(conv3x3(self.width, self.width, stride))
            self.bns.append(nn.BatchNorm2d(self.width))
        ### ===== ① HPC 模块结束 =====
        ### ===== ② SPR 模块（通道权重学习） =====
        # 对每个分支的特征使用 SPRModule 学习通道权重
        self.attention = SPRModule(self.width)
        ### ===== SPR 模块结束 =====
        ### Softmax 用于跨尺度通道权重归一化
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        feats = []
        for i in range(self.nums):
            sp = self.convs[i](x)
            sp = self.bns[i](sp)
            feats.append(sp)

        # 应用注意力机制
        attn_weight = [self.attention(feat) for feat in feats]
        attn_weight = torch.cat(attn_weight, dim=1)
        attn_vectors = attn_weight.view(batch_size, self.nums, self.width, 1, 1)
        attn_vectors = self.softmax(attn_vectors)

        # 将注意力权重应用于每个尺度的特征图
        feats_weight = [feat.unsqueeze(1) * attn_vectors[:, i:i + 1] for i, feat in enumerate(feats)]

        # 将所有尺度的特征图合并
        out = torch.cat(feats_weight, 1).sum(dim=1)

        return out

class Mix2(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix2, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = w
        self.mix_block = nn.ReLU()

    def forward(self, x):
        mix_factor = self.mix_block(self.w)
        out = x * mix_factor
        return out

class MSFM(nn.Module):
    def __init__(self, features, r=2, L=24) -> None:
        super().__init__()
        d = max(int(features / r), L)
        self.features = features

        self.convll = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU()
        )
        self.convl = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU()
        )
        self.convm = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.Sequential(
            nn.Conv2d(features, features, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(features, features, 1, 1, 0)
        )

        self.softmax = nn.Softmax(dim=1)
        self.sigmod = nn.Sigmoid()
        self.cov7 = nn.Conv2d(2, 1, kernel_size=1, bias=True)

        self.mix = Mix2(m=0.8)
        self.out = nn.Sequential(
            nn.Conv2d(features * 2, features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(features)  # 论文要求先归一化

    def forward(self, x):
        # 论文3.2节核心流程：归一化→多尺度卷积→频率融合→注意力校准
        x = self.norm(x)  # 新增BatchNorm，匹配论文流程
        lowlow = self.convll(x)
        low = self.convl(lowlow)
        middle = self.convm(low)

        emerge1 = low + lowlow + middle  # 多尺度特征融合
        emerge2 = self.gap(emerge1)
        emerge2 = self.softmax(self.fcs(emerge2))
        fea_high = emerge2 * emerge1  # 通道注意力

        # 频率特征增强（论文IFFE模块核心）
        max_out, _ = torch.max(emerge1, dim=1, keepdim=True)
        avg_out = torch.mean(emerge1, dim=1, keepdim=True)
        spa_out = self.sigmod(self.cov7(torch.cat([max_out, avg_out], dim=1)))
        fea_high2 = spa_out * emerge1  # 空间注意力

        out = self.out(torch.cat([fea_high, fea_high2], dim=1))
        return out + self.mix(x)  # 残差连接，避免信息丢失
    
class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture新增
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate, scale = 1.0):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    self.scale = scale
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out) * self.scale
    out = out + x
    return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.msfm = MSFM(features=channels)  # 替换原MSAModule，符合论文MSFM部署逻辑
        self.msam = MSAModule(inplanes=channels, scale=3, stride=1, stype='normal')  # 保留MSAModule互补
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.msfm(out)  # 第一步：MSFM多尺度+频率融合
        out = self.msam(out) * 0.1  # 第二步：MSAModule注意力校准
        return out + residual
    
class Net(nn.Module):
    def __init__(self, res_blocks=18):
        super(Net, self).__init__()

        # 编码阶段（论文要求MSFM部署在每个编码阶段核心）
        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(ResidualBlock(16), ResidualBlock(16), ResidualBlock(16))  # 已含MSFM

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv1 = RDB(16, 4, 16)
        self.fusion1 = Encoder_MDCBlock1(16, 2, mode='iter2')
        self.dense1 = nn.Sequential(ResidualBlock(32), ResidualBlock(32), ResidualBlock(32))  # 已含MSFM

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv2 = RDB(32, 4, 32)
        self.fusion2 = Encoder_MDCBlock1(32, 3, mode='iter2')
        self.dense2 = nn.Sequential(ResidualBlock(64), ResidualBlock(64), ResidualBlock(64))  # 已含MSFM

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv3 = RDB(64, 4, 64)
        self.fusion3 = Encoder_MDCBlock1(64, 4, mode='iter2')
        self.dense3 = nn.Sequential(ResidualBlock(128), ResidualBlock(128), ResidualBlock(128))  # 已含MSFM

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.conv4 = RDB(128, 4, 128)
        self.fusion4 = Encoder_MDCBlock1(128, 5, mode='iter2')

        # 瓶颈层（论文要求部署2个MSFM，增强全局特征）
        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))  # 已含MSFM

        # 解码阶段（论文要求MSFM与编码阶段对称部署）
        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(128), ResidualBlock(128), ResidualBlock(128))  # 已含MSFM
        self.conv_4 = RDB(64, 4, 64)
        self.fusion_4 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(64), ResidualBlock(64), ResidualBlock(64))  # 已含MSFM
        self.conv_3 = RDB(32, 4, 32)
        self.fusion_3 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(32), ResidualBlock(32), ResidualBlock(32))  # 已含MSFM
        self.conv_2 = RDB(16, 4, 16)
        self.fusion_2 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(ResidualBlock(16), ResidualBlock(16), ResidualBlock(16))  # 已含MSFM
        self.conv_1 = RDB(8, 4, 8)
        self.fusion_1 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

    def forward(self, x):
        res1x = self.conv_input(x)
        res1x_1, res1x_2 = res1x.split([8, 8], dim=1)
        feature_mem = [res1x_1]
        x = self.dense0(res1x) + res1x  # 编码1：含MSFM

        res2x = self.conv2x(x)
        res2x_1, res2x_2 = res2x.split([16, 16], dim=1)
        res2x_1 = self.fusion1(res2x_1, feature_mem)
        res2x_2 = self.conv1(res2x_2)
        feature_mem.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.dense1(res2x) + res2x  # 编码2：含MSFM

        res4x = self.conv4x(res2x)
        res4x_1, res4x_2 = res4x.split([32, 32], dim=1)
        res4x_1 = self.fusion2(res4x_1, feature_mem)
        res4x_2 = self.conv2(res4x_2)
        feature_mem.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.dense2(res4x) + res4x  # 编码3：含MSFM

        res8x = self.conv8x(res4x)
        res8x_1, res8x_2 = res8x.split([64, 64], dim=1)
        res8x_1 = self.fusion3(res8x_1, feature_mem)
        res8x_2 = self.conv3(res8x_2)
        feature_mem.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.dense3(res8x) + res8x  # 编码4：含MSFM

        res16x = self.conv16x(res8x)
        res16x_1, res16x_2 = res16x.split([128, 128], dim=1)
        res16x_1 = self.fusion4(res16x_1, feature_mem)
        res16x_2 = self.conv4(res16x_2)
        res16x = torch.cat((res16x_1, res16x_2), dim=1)

        # 瓶颈层：含MSFM
        res_dehaze = res16x
        in_ft = res16x * 2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze

        # 解码阶段
        res16x_1, res16x_2 = res16x.split([128, 128], dim=1)
        feature_mem_up = [res16x_1]
        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x  # 解码1：含MSFM
        res8x_1, res8x_2 = res8x.split([64, 64], dim=1)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x  # 解码2：含MSFM
        res4x_1, res4x_2 = res4x.split([32, 32], dim=1)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x  # 解码3：含MSFM
        res2x_1, res2x_2 = res2x.split([16, 16], dim=1)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x  # 解码4：含MSFM
        x_1, x_2 = x.split([8, 8], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)
        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        return self.conv_output(x)