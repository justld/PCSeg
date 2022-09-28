import math
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

BatchNorm2D = nn.BatchNorm2D
bn_mom = 0.9


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2D(planes, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2D(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Layer):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1)
        self.bn1 = BatchNorm2D(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm2D(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2D(
            planes,
            planes * self.expansion,
            kernel_size=1, )
        self.bn3 = BatchNorm2D(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=5, stride=2, padding=2),
            BatchNorm2D(
                inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale2 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=9, stride=4, padding=4),
            BatchNorm2D(
                inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale3 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=17, stride=8, padding=8),
            BatchNorm2D(
                inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        # self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2D((1, 1)),
        #                             BatchNorm2D(inplanes, momentum=bn_mom),
        #                             nn.ReLU(),
        #                             nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
        #                             )
        self.scale4 = nn.Sequential(
            BatchNorm2D(
                inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale0 = nn.Sequential(
            BatchNorm2D(
                inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.process1 = nn.Sequential(
            BatchNorm2D(
                branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process2 = nn.Sequential(
            BatchNorm2D(
                branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process3 = nn.Sequential(
            BatchNorm2D(
                branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process4 = nn.Sequential(
            BatchNorm2D(
                branch_planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.compression = nn.Sequential(
            BatchNorm2D(
                branch_planes * 5, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes * 5, outplanes, kernel_size=1, bias_attr=False), )
        self.shortcut = nn.Sequential(
            BatchNorm2D(
                inplanes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, outplanes, kernel_size=1, bias_attr=False), )

    def forward(self, x):
        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((F.interpolate(
                self.scale1(x),
                size=[height, width],
                mode='bilinear',
                align_corners=True) + x_list[0])))
        x_list.append((self.process2((F.interpolate(
            self.scale2(x),
            size=[height, width],
            mode='bilinear',
            align_corners=True) + x_list[1]))))
        x_list.append(
            self.process3((F.interpolate(
                self.scale3(x),
                size=[height, width],
                mode='bilinear',
                align_corners=True) + x_list[2])))
        x_list.append(
            self.process4((F.interpolate(
                self.scale4(x),
                size=[height, width],
                mode='bilinear',
                align_corners=True) + x_list[3])))

        out = self.compression(paddle.concat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2D(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2D(
            inplanes, interplanes, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = BatchNorm2D(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            interplanes, outplanes, kernel_size=1, padding=0, bias_attr=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(
                out, size=[height, width], mode='bilinear', align_corners=True)

        return out


class DualResNet(nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 in_channels=5,
                 num_classes=19,
                 planes=64,
                 spp_planes=128,
                 head_planes=128,
                 augment=False):
        super(DualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment

        self.conv1 = nn.Sequential(
            nn.Conv2D(
                in_channels, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2D(
                planes, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2D(
                planes, momentum=bn_mom),
            nn.ReLU(), )

        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(
            block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2D(
                planes * 4, highres_planes, kernel_size=1, bias_attr=False),
            BatchNorm2D(
                highres_planes, momentum=bn_mom), )

        self.compression4 = nn.Sequential(
            nn.Conv2D(
                planes * 8, highres_planes, kernel_size=1, bias_attr=False),
            BatchNorm2D(
                highres_planes, momentum=bn_mom), )

        self.down3 = nn.Sequential(
            nn.Conv2D(
                highres_planes,
                planes * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False),
            BatchNorm2D(
                planes * 4, momentum=bn_mom), )

        self.down4 = nn.Sequential(
            nn.Conv2D(
                highres_planes,
                planes * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False),
            BatchNorm2D(
                planes * 4, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                planes * 4,
                planes * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False),
            BatchNorm2D(
                planes * 8, momentum=bn_mom), )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes,
                                        2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes,
                                        highres_planes, 1)

        self.layer5 = self._make_layer(
            Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes,
                                             num_classes)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()(m.weight)
            elif isinstance(m, BatchNorm2D):
                nn.initializer.Constant(value=1)(m.weight)
                nn.initializer.Constant(value=0)(m.bias)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                nn.BatchNorm2D(
                    planes * block.expansion, momentum=bn_mom), )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        N, C, H, W = x.shape
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)

        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3(self.relu(layers[2])),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=True)
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=True)

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=True)

        x_ = self.final_layer(x + x_)
        logits = [x_]
        if self.training and self.augment:
            x_extra = self.seghead_extra(temp)
            logits.append(x_extra)
        return [
            F.interpolate(
                logit, size=(H, W), mode='bilinear', align_corners=True)
            for logit in logits
        ]


def DDRNet23_slim(in_channels, num_classes):
    model = DualResNet(
        BasicBlock, [2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        planes=32,
        spp_planes=128,
        head_planes=64,
        augment=True)
    return model


if __name__ == '__main__':
    x = paddle.rand([1, 5, 64, 2048])
    model = DDRNet23_slim(in_channels=5, num_classes=20)
    y = model(x)
    for t in y:
        print(t.shape)
