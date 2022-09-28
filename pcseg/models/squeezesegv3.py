from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class SqueezeSegV3(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels=5,
                 layers=21,
                 drop_prob=0.01,
                 bn_d=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(
            in_channels=in_channels,
            drop_prob=drop_prob,
            layers=layers,
            output_stride=8,
            bn_d=0.01)
        self.decoder = Decoder(
            bn_d=bn_d,
            drop_prob=drop_prob,
            feature_depth=self.encoder.feat_channels,
            OS=self.encoder.OS)
        self.head1 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(
                256, self.num_classes, kernel_size=1, stride=1, padding=0))

        self.head2 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(
                256, self.num_classes, kernel_size=1, stride=1, padding=0))

        self.head3 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(
                128, self.num_classes, kernel_size=1, stride=1, padding=0))

        self.head4 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(
                64, self.num_classes, kernel_size=1, stride=1, padding=0))
        self.head5 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(
                32, self.num_classes, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        feature, skips = self.encoder(x)
        y = self.decoder(feature, skips)
        z1 = self.head5(y[0])
        z2 = self.head4(y[1])
        z3 = self.head3(y[2])
        z4 = self.head2(y[3])
        z5 = self.head1(y[4])
        if not self.training:
            return [z1, z2, z3, z4, z5]
        else:
            return [z1]


class Decoder(nn.Layer):
    """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" Layer
  """

    def __init__(self, bn_d=0.01, drop_prob=0.01, OS=32, feature_depth=1024):
        super(Decoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = drop_prob
        self.bn_d = bn_d

        # stride play
        self.strides = [1, 1, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)

        # decoder
        self.dec5 = self._make_dec_layer(
            BasicBlock, [self.backbone_feature_depth, 256],
            bn_d=self.bn_d,
            stride=self.strides[0])
        self.dec4 = self._make_dec_layer(
            BasicBlock, [256, 256], bn_d=self.bn_d, stride=self.strides[1])
        self.dec3 = self._make_dec_layer(
            BasicBlock, [256, 128], bn_d=self.bn_d, stride=self.strides[2])
        self.dec2 = self._make_dec_layer(
            BasicBlock, [128, 64], bn_d=self.bn_d, stride=self.strides[3])
        self.dec1 = self._make_dec_layer(
            BasicBlock, [64, 32], bn_d=self.bn_d, stride=self.strides[4])

        # layer list to execute with skips
        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]

        # for a bit of fun
        self.dropout = nn.Dropout2D(self.drop_prob)

        # last channels
        self.last_channels = 32

    def _make_dec_layer(self, block, planes, bn_d=0.9, stride=2, flag=True):
        layers = []

        #  downsample
        if stride == 2:
            layers.append(
                nn.Conv2DTranspose(
                    planes[0],
                    planes[1],
                    kernel_size=[1, 4],
                    stride=[1, 2],
                    padding=[0, 1]))
        else:
            layers.append(
                nn.Conv2D(
                    planes[0], planes[1], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2D(planes[1], momentum=bn_d))
        layers.append(nn.LeakyReLU(0.1))

        #  blocks
        layers.append(block(planes[1], planes, bn_d))

        return nn.Sequential(*layers)

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2  # match skip
            feats = feats + skips[os].detach()  # add skip
        x = feats
        return x, skips, os

    def forward(self, x, skips):
        os = self.backbone_OS

        # run layers
        x1, skips, os = self.run_layer(x, self.dec5, skips, os)
        x2, skips, os = self.run_layer(x1, self.dec4, skips, os)
        x3, skips, os = self.run_layer(x2, self.dec3, skips, os)
        x4, skips, os = self.run_layer(x3, self.dec2, skips, os)
        x5, skips, os = self.run_layer(x4, self.dec1, skips, os)

        x5 = self.dropout(x5)

        return [x5, x4, x3, x2, x1]


class BasicBlock(nn.Layer):
    def __init__(self, inplanes, planes, bn_d=0.9):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(
            inplanes,
            planes[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2D(
            planes[0],
            planes[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class Encoder(nn.Layer):
    def __init__(self,
                 in_channels=5,
                 drop_prob=0.01,
                 layers=21,
                 output_stride=8,
                 bn_d=0.9):
        super().__init__()
        self.layers = layers
        self.drop_prob = drop_prob
        self.bn_d = bn_d
        self.strides = [2, 2, 2, 1, 1]
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)
        self.OS = output_stride
        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)
        else:

            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)

        assert self.layers in model_blocks.keys()

        self.blocks = model_blocks[self.layers]

        self.conv1 = nn.Conv2D(
            in_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        self.enc1 = self._make_enc_layer(
            SACBlock, [32, 64],
            self.blocks[0],
            stride=self.strides[0],
            DS=True,
            bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(
            SACBlock, [64, 128],
            self.blocks[1],
            stride=self.strides[1],
            DS=True,
            bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(
            SACBlock, [128, 256],
            self.blocks[2],
            stride=self.strides[2],
            DS=True,
            bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(
            SACBlock, [256, 256],
            self.blocks[3],
            stride=self.strides[3],
            DS=False,
            bn_d=self.bn_d)
        self.enc5 = self._make_enc_layer(
            SACBlock, [256, 256],
            self.blocks[4],
            stride=self.strides[4],
            DS=False,
            bn_d=self.bn_d)

        self.dropout = nn.Dropout2D(self.drop_prob)

        self.feat_channels = 256

    def _make_enc_layer(self, block, planes, blocks, stride, DS, bn_d=0.9):
        layers = []

        inplanes = planes[0]
        for i in range(0, blocks):
            layers.append(block(inplanes, planes, bn_d))
        if DS:
            layers.append(
                nn.Conv2D(
                    planes[0],
                    planes[1],
                    kernel_size=3,
                    stride=[1, stride],
                    dilation=1,
                    padding=1,
                    bias_attr=False))
            layers.append(nn.BatchNorm2D(planes[1], momentum=bn_d))
            layers.append(nn.LeakyReLU(0.1))

        return nn.Sequential(*layers)

    def run_layer(self, xyz, feature, layer, skips, os, flag=True):
        new_xyz = xyz
        if flag:
            xyz, new_xyz, y = layer[:-3]([xyz, new_xyz, feature])
            y = layer[-3:](y)
            xyz = F.interpolate(xyz, size=[xyz.shape[2], xyz.shape[3] // 2])
        else:
            xyz, new_xyz, y = layer([xyz, new_xyz, feature])
        if y.shape[2] < feature.shape[2] or y.shape[3] < feature.shape[3]:
            skips[os] = feature.detach()
            os *= 2
        feature = self.dropout(y)
        return xyz, feature, skips, os

    def forward(self, feature):
        skips = {}
        os = 1
        xyz = feature[:, 1:4, :, :]
        feature = self.relu1(self.bn1(self.conv1(feature)))

        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc1, skips,
                                                 os)
        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc2, skips,
                                                 os)
        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc3, skips,
                                                 os)
        xyz, feature, skips, os = self.run_layer(
            xyz, feature, self.enc4, skips, os, flag=False)
        xyz, feature, skips, os = self.run_layer(
            xyz, feature, self.enc5, skips, os, flag=False)

        return feature, skips


class SACBlock(nn.Layer):
    def __init__(self, inplanes, expand1x1_planes, bn_d=0.1):
        super(SACBlock, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d

        self.attention_x = nn.Sequential(
            nn.Conv2D(
                3, 9 * self.inplanes, kernel_size=7, padding=3),
            nn.BatchNorm2D(
                9 * self.inplanes, momentum=0.9), )

        self.position_mlp_2 = nn.Sequential(
            nn.Conv2D(
                9 * self.inplanes, self.inplanes, kernel_size=1),
            nn.BatchNorm2D(
                self.inplanes, momentum=0.9),
            nn.ReLU(),
            nn.Conv2D(
                self.inplanes, self.inplanes, kernel_size=3, padding=1),
            nn.BatchNorm2D(
                self.inplanes, momentum=0.9),
            nn.ReLU(), )

    def forward(self, input):
        xyz = input[0]
        new_xyz = input[1]
        feature = input[2]
        N, C, H, W = feature.shape

        new_feature = F.unfold(
            feature, kernel_sizes=3, paddings=1).reshape([N, -1, H, W])
        attention = F.sigmoid(self.attention_x(new_xyz))
        new_feature = new_feature * attention
        new_feature = self.position_mlp_2(new_feature)
        fuse_feature = new_feature + feature

        return xyz, new_xyz, fuse_feature


if __name__ == '__main__':
    x = paddle.rand([1, 5, 64, 1024])
    model = SqueezeSegV3(in_channels=5, num_classes=20, layers=21)
    y = model(x)
    for t in y:
        print(t.numpy().shape)
