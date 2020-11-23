import torch
import torch.nn as nn
import math

from config import device, im_size

class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
            with_relu=True
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown1(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown1, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp1(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp1, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv = conv2DBatchNormRelu(in_size, out_size, k_size=5, stride=1, padding=2, with_relu=False)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv(outputs)
        return outputs


class DIMModel(nn.Module):
    def __init__(self, n_classes=1, in_channels=4, is_unpooling=True, pretrain=True):
        super(DIMModel, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.pretrain = pretrain

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp1(512, 512)
        self.up4 = segnetUp1(512, 256)
        self.up3 = segnetUp1(256, 128)
        self.up2 = segnetUp1(128, 64)
        self.up1 = segnetUp1(64, n_classes)

        self.sigmoid = nn.Sigmoid()

        if self.pretrain:
            import torchvision.models as models
            vgg16 = models.vgg16()
            self.init_vgg16_params(vgg16)

    def forward(self, inputs):
        # inputs: [N, 4, 320, 320]
        feature_maps = []
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        # decoder features
        feature_maps.append(down1)
        feature_maps.append(down2)
        feature_maps.append(down3)
        feature_maps.append(down4)
        feature_maps.append(down5)

        # encoder features
        feature_maps.append(up5)
        feature_maps.append(up4)
        feature_maps.append(up3)
        feature_maps.append(up2)

        x = torch.squeeze(up1, dim=1)  # [N, 1, 320, 320] -> [N, 320, 320]
        x = self.sigmoid(x)

        return feature_maps, x

    def extract_feature(self, inputs):
        feature_maps = []
        down1, indices_1, unpool_shape1 = self.down1(inputs)

        down2 = self.down2.conv1(down1)
        down2 = self.down2.conv2.cbr_unit[0](down2) #conv
        feature1 = self.down2.conv2.cbr_unit[1](down2) #bn
        down2 = self.down2.conv2.cbr_unit[2](feature1) #relu
        down2, indices_2 = self.down2.maxpool_with_argmax(down2)

        down3 = self.down3.conv1(down2)
        down3 = self.down3.conv2(down3)
        down3 = self.down3.conv3.cbr_unit[0](down3) #conv
        feature2 = self.down3.conv3.cbr_unit[1](down3) #bn
        down3 = self.down3.conv3.cbr_unit[2](feature2) #relu
        down3, indices_3 = self.down3.maxpool_with_argmax(down3)

        down4 = self.down4.conv1(down3)
        down4 = self.down4.conv2(down4)
        down4 = self.down4.conv3.cbr_unit[0](down4) #conv
        feature3 = self.down4.conv3.cbr_unit[1](down4) #bn
        down4 = self.down4.conv3.cbr_unit[2](feature3) #relu
        down4, indices_4 = self.down4.maxpool_with_argmax(down4)

        down5 = self.down5.conv1(down4)
        down5 = self.down5.conv2(down5)
        down5 = self.down5.conv3.cbr_unit[0](down5) #conv
        feature4 = self.down5.conv3.cbr_unit[1](down5) #bn
        down5 = self.down5.conv3.cbr_unit[2](feature4) #relu
        down5, indices_5 = self.down5.maxpool_with_argmax(down5)

        up5 = self.up5(down5, indices_5, feature4.size())
        up4 = self.up4(up5, indices_4, feature3.size())
        up3 = self.up3(up4, indices_3, feature2.size())
        up2 = self.up2(up3, indices_2, feature1.size())
        up1 = self.up1(up2, indices_1, unpool_shape1)

        feature1, _ = self.down2.maxpool_with_argmax(feature1)
        feature2, _ = self.down2.maxpool_with_argmax(feature2)
        feature3, _ = self.down2.maxpool_with_argmax(feature3)
        feature4, _ = self.down2.maxpool_with_argmax(feature4)

        feature_maps.append(feature1)
        feature_maps.append(feature2)
        feature_maps.append(feature3)
        feature_maps.append(feature4)

        x = torch.squeeze(up1, dim=1)  # [N, 1, 320, 320] -> [N, 320, 320]
        x = self.sigmoid(x)

        return feature_maps, x

    def get_bn_before_relu(self):
        bn1 = self.down2.conv2.cbr_unit[1]
        bn2 = self.down3.conv3.cbr_unit[1]
        bn3 = self.down4.conv3.cbr_unit[1]
        bn4 = self.down5.conv3.cbr_unit[1]

        return [bn1, bn2, bn3, bn4]

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.weight.size() == l2.weight.size() and l1.bias.size() == l2.bias.size():
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

class DIMModel_student(nn.Module):
    def __init__(self, n_classes=1, in_channels=4, is_unpooling=True, n_features=32):
        super(DIMModel_student, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.n_features = n_features

        self.down1 = segnetDown1(self.in_channels, self.n_features) # 4, 32
        self.down2 = segnetDown1(self.n_features, self.n_features*2) # 32, 64
        self.down3 = segnetDown1(self.n_features*2, self.n_features*4) # 64, 128
        self.down4 = segnetDown2(self.n_features*4, self.n_features*8) # 128, 256
        self.down5 = segnetDown2(self.n_features*8, self.n_features*8) # 256, 256

        self.up5 = segnetUp1(self.n_features*8, self.n_features*8) # 256, 256
        self.up4 = segnetUp1(self.n_features*8, self.n_features*4) # 256, 128
        self.up3 = segnetUp1(self.n_features*4, self.n_features*2) # 128, 64
        self.up2 = segnetUp1(self.n_features*2, self.n_features) # 64, 32
        self.up1 = segnetUp1(self.n_features, n_classes)

        self.sigmoid = nn.Sigmoid()
        # for channel similarity
        self.feature1 = nn.Conv2d(n_features, 64, kernel_size=1, stride=1, padding=0)
        self.feature2 = nn.Conv2d(n_features*2, 128, kernel_size=1, stride=1, padding=0)
        self.feature3 = nn.Conv2d(n_features*4, 256, kernel_size=1, stride=1, padding=0)
        self.feature4 = nn.Conv2d(n_features*8, 512, kernel_size=1, stride=1, padding=0)
        self.feature5 = nn.Conv2d(n_features*8, 512, kernel_size=1, stride=1, padding=0)

        self.feature6 = nn.Conv2d(n_features*8, 512, kernel_size=1, stride=1, padding=0)
        self.feature7 = nn.Conv2d(n_features*4, 256, kernel_size=1, stride=1, padding=0)
        self.feature8 = nn.Conv2d(n_features*2, 128, kernel_size=1, stride=1, padding=0)
        self.feature9 = nn.Conv2d(n_features, 64, kernel_size=1, stride=1, padding=0)

        # OFD
        self.feature_trans2 = self.build_feature_connector(128, n_features*2)
        self.feature_trans3 = self.build_feature_connector(256, n_features*4)
        self.feature_trans4 = self.build_feature_connector(512, n_features*8)
        self.feature_trans5 = self.build_feature_connector(512, n_features*8)

    def forward(self, inputs):
        # inputs: [N, 4, 320, 320]
        feature_maps = []

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        # encoder features
        feature_maps.append(down1)
        feature_maps.append(down2)
        feature_maps.append(down3)
        feature_maps.append(down4)
        feature_maps.append(down5)

        # decoder features
        feature_maps.append(up5)
        feature_maps.append(up4)
        feature_maps.append(up3)
        feature_maps.append(up2)

        # for channel similarity
        down1 = self.feature1(down1)
        down2 = self.feature2(down2)
        down3 = self.feature3(down3)
        down4 = self.feature4(down4)
        down5 = self.feature5(down5)

        up5 = self.feature6(up5)
        up4 = self.feature7(up4)
        up3 = self.feature8(up3)
        up2 = self.feature9(up2)

        feature_maps.append(down1)
        feature_maps.append(down2)
        feature_maps.append(down3)
        feature_maps.append(down4)
        feature_maps.append(down5)

        feature_maps.append(up5)
        feature_maps.append(up4)
        feature_maps.append(up3)
        feature_maps.append(up2)


        x = torch.squeeze(up1, dim=1)  # [N, 1, 320, 320] -> [N, 320, 320]
        x = self.sigmoid(x)

        return feature_maps, x

    def extract_feature(self, inputs):
        # inputs: [N, 4, 320, 320]
        feature_maps = []

        down1, indices_1, unpool_shape1 = self.down1(inputs)

        down2 = self.down2.conv1.cbr_unit[0](down1)  # conv
        feature1 = self.down2.conv1.cbr_unit[1](down2)  # bn
        down2 = self.down2.conv1.cbr_unit[2](feature1)  # relu
        down2, indices_2 = self.down2.maxpool_with_argmax(down2)

        down3 = self.down3.conv1.cbr_unit[0](down2)  # conv
        feature2 = self.down3.conv1.cbr_unit[1](down3)  # bn
        down3 = self.down3.conv1.cbr_unit[2](feature2)  # relu
        down3, indices_3 = self.down3.maxpool_with_argmax(down3)

        down4 = self.down4.conv1(down3)
        down4 = self.down4.conv2.cbr_unit[0](down4)  # conv
        feature3 = self.down4.conv2.cbr_unit[1](down4)  # bn
        down4 = self.down4.conv2.cbr_unit[2](feature3)  # relu
        down4, indices_4 = self.down4.maxpool_with_argmax(down4)

        down5 = self.down5.conv1(down4)
        down5 = self.down5.conv2.cbr_unit[0](down5)  # conv
        feature4 = self.down5.conv2.cbr_unit[1](down5)  # bn
        down5 = self.down5.conv2.cbr_unit[2](feature4)  # relu
        down5, indices_5 = self.down5.maxpool_with_argmax(down5)

        up5 = self.up5(down5, indices_5, feature4.size())
        up4 = self.up4(up5, indices_4, feature3.size())
        up3 = self.up3(up4, indices_3, feature2.size())
        up2 = self.up2(up3, indices_2, feature1.size())
        up1 = self.up1(up2, indices_1, unpool_shape1)

        feature1, _ = self.down2.maxpool_with_argmax(feature1)
        feature2, _ = self.down2.maxpool_with_argmax(feature2)
        feature3, _ = self.down2.maxpool_with_argmax(feature3)
        feature4, _ = self.down2.maxpool_with_argmax(feature4)

        feature1 = self.feature_trans2(feature1)
        feature2 = self.feature_trans3(feature2)
        feature3 = self.feature_trans4(feature3)
        feature4 = self.feature_trans5(feature4)

        feature_maps.append(feature1)
        feature_maps.append(feature2)
        feature_maps.append(feature3)
        feature_maps.append(feature4)

        x = torch.squeeze(up1, dim=1)  # [N, 1, 320, 320] -> [N, 320, 320]
        x = self.sigmoid(x)

        return feature_maps, x

    def build_feature_connector(self, t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel)]

        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        return nn.Sequential(*C)
if __name__ == '__main__':
    model = DIMModel.to(device)
    #summary(model, (4, im_size, im_size))

