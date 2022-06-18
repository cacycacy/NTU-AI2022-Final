import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MLP(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio

        self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Channels_Attention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(Channels_Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = MLP(channels=channels, reduction_ratio=reduction_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.sigmoid(att)
        att = att.unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * att
        return x

class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att1 = x.max(dim=1)[0].unsqueeze(1)
        att2 = x.mean(dim=1).unsqueeze(1)
        att = torch.cat((att1, att2), dim=1)
        att = self.conv(att)
        att = self.bn(att)
        att = self.sigmoid(att)
        x = x * att
        return x

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.ch = Channels_Attention(channels=channels, reduction_ratio=reduction_ratio)
        self.sp = Spatial_Attention()

    def forward(self, x):
        x = self.ch(x)
        x = self.sp(x)
        return x

class ConvBnPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super(ConvBnPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              groups=groups, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DepthWise(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1, residual=False):
        super(DepthWise, self).__init__()
        self.conv = ConvBnPReLU(in_channels, groups, kernel_size=1, stride=1, padding=0)
        self.dwconv = ConvBnPReLU(groups, groups, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.attention_module = CBAMBlock(groups)
        self.project = ConvBn(groups, out_channels, kernel_size=1, padding=0, stride=1)
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.dwconv(x)
        x = self.attention_module(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class MultiDepthWiseRes(nn.Module):
    def __init__(self, channels, num_block, groups, kernel_size=3, stride=1, padding=1):
        super(MultiDepthWiseRes, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, groups=groups, residual=True))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class GDC(nn.Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.conv = ConvBn(768, 768, kernel_size=7, stride=1, padding=0, groups=768)
        self.flatten = Flatten()
        self.linear = nn.Linear(768, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x

class SeesawFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(SeesawFaceNet, self).__init__()
        self.conv1 = ConvBnPReLU(3, 92, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBnPReLU(92, 92, kernel_size=3, stride=1, padding=1, groups=92)
        self.conv3 = DepthWise(92, 92, kernel_size=3, stride=2, padding=1, groups=192)
        self.conv4 = MultiDepthWiseRes(92, num_block=4, kernel_size=3, stride=1, padding=1, groups=192)
        self.conv5 = DepthWise(92, 192, kernel_size=3, stride=2, padding=1, groups=384)
        self.conv6 = MultiDepthWiseRes(192, num_block=6, kernel_size=3, stride=1, padding=1, groups=384)
        self.conv7 = DepthWise(192, 192, kernel_size=3, stride=2, padding=1, groups=768)
        self.conv8 = MultiDepthWiseRes(192, num_block=2, kernel_size=3, stride=1, padding=1, groups=384)
        self.conv9 = ConvBnPReLU(192, 768, kernel_size=1, stride=1, padding=0)
        self.output_layer = GDC(embedding_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.output_layer(out)
        return out

    # def visualize_hm(self, x):
    #     out = self.conv1(x)
    #     out = self.conv2(out)
    #     out = self.conv3(out)
    #     out = self.conv4(out)
    #     out = self.conv5(out)
    #     out = self.conv6(out)
    #     out = self.conv7(out)
    #     out = self.conv8(out)
    #     out = self.conv9(out)
    #     out_9 = out
    #     out = self.output_layer(out)
    #     return out, out_9

    def visualize_fm(self, x):
        out = self.conv1(x)
        out_1 = out
        out = self.conv2(out)
        out_2 = out
        out = self.conv3(out)
        out_3 = out
        out = self.conv4(out)
        out_4 = out
        out = self.conv5(out)
        out_5 = out
        out = self.conv6(out)
        out_6 = out
        out = self.conv7(out)
        out_7 = out
        out = self.conv8(out)
        out_8 = out
        out = self.conv9(out)
        out_9 = out
        out = self.output_layer(out)
        return out, [out_9, out_8, out_7, out_6, out_5, out_4, out_3, out_2, out_1]

def get_parameters():
    model = SeesawFaceNet()
    total = sum([para.numel() for para in model.parameters()])
    print(f"Total parameters:{total}")

def seperate_model_bn(model):

    modules = [*model.named_parameters()]
    modules_wo_bn = []

    modules_bn = []
    for module in modules:
        if 'bn' in module[0]:
            modules_bn.extend([module[1]])
        else:
            modules_wo_bn.extend([module[1]])
    return modules_wo_bn, modules_bn


