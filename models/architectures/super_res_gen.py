import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# @torch.no_grad()
# def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
#     """Initialize network weights.
#     Args:
#         module_list (list[nn.Module] | nn.Module): Modules to be initialized.
#         scale (float): Scale initialized weights, especially for residual
#             blocks. Default: 1.
#         bias_fill (float): The value to fill bias. Default: 0
#         kwargs (dict): Other arguments for initialization function.
#     """
#     if not isinstance(module_list, list):
#         module_list = [module_list]
#     for module in module_list:
#         for m in module.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, **kwargs)
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, **kwargs)
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)
#             elif isinstance(m, _BatchNorm):
#                 init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)


# def make_layer(basic_block, num_basic_block, **kwarg):
#     """Make layers by stacking the same blocks.
#     Args:
#         basic_block (nn.module): nn.module class for basic block.
#         num_basic_block (int): number of blocks.
#     Returns:
#         nn.Sequential: Stacked blocks in nn.Sequential.
#     """
#     layers = []
#     for _ in range(num_basic_block):
#         layers.append(basic_block(**kwarg))
#     return nn.Sequential(*layers)


# def pixel_unshuffle(x, scale):
#     """ Pixel unshuffle.
#     Args:
#         x (Tensor): Input feature with shape (b, c, hh, hw).
#         scale (int): Downsample ratio.
#     Returns:
#         Tensor: the pixel unshuffled feature.
#     """
#     b, c, hh, hw = x.size()
#     out_channel = c * (scale ** 2)
#     assert hh % scale == 0 and hw % scale == 0
#     h = hh // scale
#     w = hw // scale
#     x_view = x.view(b, c, h, scale, w, scale)
#     return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class ResidualDenseBlock(nn.Module):
#     """Residual Dense Block.
#     Used in RRDB block in ESRGAN.
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         num_grow_ch (int): Channels for each growth.
#     """
#
#     def __init__(self, num_feat=64, num_grow_ch=32):
#         super(ResidualDenseBlock, self).__init__()
#         self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
#         self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
#         self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
#         self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
#         self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#         # initialization
#         default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
#
#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         # Empirically, we use 0.2 to scale the residual for better performance
#         return x5 * 0.2 + x


# class RRDB(nn.Module):
#     """Residual in Residual Dense Block.
#     Used in RRDB-Net in ESRGAN.
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         num_grow_ch (int): Channels for each growth.
#     """
#
#     def __init__(self, num_feat, num_grow_ch=32):
#         super(RRDB, self).__init__()
#         self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
#         self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
#         self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
#
#     def forward(self, x):
#         out = self.rdb1(x)
#         out = self.rdb2(out)
#         out = self.rdb3(out)
#         # Empirically, we use 0.2 to scale the residual for better performance
#         return out * 0.2 + x


# class RRDBNet(nn.Module):
#     """Networks consisting of Residual in Residual Dense Block, which is used
#     in ESRGAN.
#     ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
#     We extend ESRGAN for scale x2 and scale x1.
#     Note: This is one option for scale 1, scale 2 in RRDBNet.
#     We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
#     and enlarge the channel size before feeding inputs into the main ESRGAN architecture.
#     Args:
#         num_in_ch (int): Channel number of inputs.
#         num_out_ch (int): Channel number of outputs.
#         num_feat (int): Channel number of intermediate features.
#             Default: 64
#         num_block (int): Block number in the trunk network. Defaults: 23
#         num_grow_ch (int): Channels for each growth. Default: 32.
#     """
#
#     def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
#         super(RRDBNet, self).__init__()
#         self.scale = scale
#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
#         self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         # upsample
#         self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#     def forward(self, x, noise, xlf):
#         feat = torch.cat([x, noise], dim=1)
#         feat = self.conv_first(feat)
#         body_feat = self.conv_body(self.body(feat))
#         feat = feat + body_feat
#         # upsample
#         feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='bicubic')))
#         feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
#         out = self.conv_last(self.lrelu(self.conv_hr(feat))).clamp(min=0, max=1)
#         return out + xlf

def initialize_weights(net_l, scale=1.0):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class SFTLayer(nn.Module):
    def __init__(self, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, nf, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, nf, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.sft0 = SFTLayer(64)
        self.sft1 = SFTLayer(32)
        self.sft2 = SFTLayer(32)
        self.sft3 = SFTLayer(32)
        self.sft4 = SFTLayer(32)
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x0_sft = self.sft0((x[0], x[1]))
        x1 = self.lrelu(self.conv1(x0_sft))
        x1_sft = self.sft1((x1, x[1]))
        x2 = self.lrelu(self.conv2(torch.cat((x[0], x1_sft), 1)))
        x2_sft = self.sft2((x2, x[1]))
        x3 = self.lrelu(self.conv3(torch.cat((x[0], x1_sft, x2_sft), 1)))
        x3_sft = self.sft3((x3, x[1]))
        x4 = self.lrelu(self.conv4(torch.cat((x[0], x1_sft, x2_sft, x3_sft), 1)))
        x4_sft = self.sft4((x4, x[1]))
        x5 = self.conv5(torch.cat((x[0], x1_sft, x2_sft, x3_sft, x4_sft), 1))
        return x5 * 0.2 + x[0]

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1((x[0],x[1]))
        out = self.RDB2((out,x[1]))
        out = self.RDB3((out,x[1]))
        return (out * 0.2 + x[0], x[1])


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf=64, nb=23, upscale=4, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upscale = upscale

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.upscale == 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.CondNet = nn.Sequential(nn.Conv2d(1, 1, 1, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(1, 32, 1))

    def forward(self, x, noise, xlf, cond):
        cond = self.CondNet(cond)
        fea = self.conv_first(torch.cat([x, noise], dim=1))
        fea2 = (fea, cond)
        fea3 = self.RRDB_trunk(fea2)
        trunk = self.trunk_conv(fea3[0])
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='bicubic')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='bicubic')))
        if self.upscale == 8:
            fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='bicubic')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out + xlf
