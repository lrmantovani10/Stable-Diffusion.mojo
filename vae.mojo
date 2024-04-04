from helpers.utils import *
from helpers.attention import *


struct Attention_Block:
    var group_norm: GroupNorm
    var attention: Self_Attention

    fn __init__(inout self, channels: Int):
        self.group_norm = GroupNorm(32, channels)
        self.attention = Self_Attention(1, channels)

    fn __copyinit__(inout self, other: Self):
        self.group_norm = other.group_norm
        self.attention = other.attention

    fn forward(inout self, inout x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var residue = x
        var out = self.group_norm.forward(x)
        out = out.reshape(x.dim0, x.dim1 * x.dim2, 1)
        out = out.transpose(0, 2)
        out = self.attention.forward(out)
        out = out.transpose(1, 2)
        out = out.reshape(x.dim0, x.dim1, x.dim2)
        out = out + residue

        return out


struct Res_Block:
    var in_channels: Int
    var out_channels: Int
    var group_norm1: GroupNorm
    var group_norm2: GroupNorm
    var conv1: Conv2D
    var conv2: Conv2D
    var res_conv_layer: Conv2D

    fn __init__(inout self, in_channels: Int, out_channels: Int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_norm1 = GroupNorm(16, in_channels)
        self.group_norm2 = GroupNorm(16, out_channels)
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, padding=(1, 1))
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=(1, 1))
        self.res_conv_layer = Conv2D(in_channels, out_channels, kernel_size=1)

    fn __copyinit__(inout self, other: Self):
        self.in_channels = other.in_channels
        self.out_channels = other.out_channels
        self.group_norm1 = other.group_norm1
        self.group_norm2 = other.group_norm2
        self.conv1 = other.conv1
        self.conv2 = other.conv2
        self.res_conv_layer = other.res_conv_layer

    fn forward(self, inout x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var residue = x
        var out = self.group_norm1.forward(x)
        out = SiLU().forward(out)
        out = self.conv1.forward(out)
        out = self.group_norm2.forward(out)
        out = SiLU().forward(out)
        out = self.conv2.forward(out)
        if self.in_channels != self.out_channels:
            residue = self.res_conv_layer.forward(residue)
        return out + residue


struct Encoder:
    var l1: Conv2D
    var l2: Res_Block
    var l3: Res_Block
    var l4: Conv2D
    var l5: Res_Block
    var l6: Res_Block
    var l7: Conv2D
    var l8: Res_Block
    var l9: Res_Block
    var l10: Conv2D
    var l11: Res_Block
    var l12: Res_Block
    var l13: Res_Block
    var l14: Attention_Block
    var l15: Res_Block
    var l16: GroupNorm
    var l17: SiLU
    var l18: Conv2D
    var l19: Conv2D

    fn __init__(
        inout self,
    ):
        self.l1 = Conv2D(3, 128, kernel_size=3, padding=(1, 1))
        self.l2 = Res_Block(128, 128)
        self.l3 = Res_Block(128, 128)
        self.l4 = Conv2D(128, 128, kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.l5 = Res_Block(128, 256)
        self.l6 = Res_Block(256, 256)
        self.l7 = Conv2D(256, 256, kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.l8 = Res_Block(256, 512)
        self.l9 = Res_Block(512, 512)
        self.l10 = Conv2D(512, 512, kernel_size=3, stride=(2, 2), padding=(0, 0))
        self.l11 = Res_Block(512, 512)
        self.l12 = Res_Block(512, 512)
        self.l13 = Res_Block(512, 512)
        self.l14 = Attention_Block(512)
        self.l15 = Res_Block(512, 512)
        self.l16 = GroupNorm(32, 512)
        self.l17 = SiLU()
        self.l18 = Conv2D(512, 8, kernel_size=3, padding=(1,1))
        self.l19 = Conv2D(8, 8, kernel_size=1, padding=(0,0))


    fn two_stride_pad(self, matrix: Matrix[float_dtype]) -> Matrix[float_dtype]:
        return matrix.pad((0, 1), (0, 1))

    fn metrics_evals(
        self, matrix: Matrix[float_dtype], noise: Matrix[float_dtype]
    ) -> Matrix[float_dtype]:
        var chunks = matrix.chunk(0, 2)
        var mean = chunks[0]
        var log_variance = chunks[1]
        log_variance = log_variance.clamp(-30, 20)
        var variance = log_variance.exp()
        var std = variance.sqrt()
        var out = mean + (noise.multiply(std))
        out *= 0.18215
        return out

    fn forward(
        inout self, x: Matrix[float_dtype], noise: Matrix[float_dtype]
    ) -> Matrix[float_dtype]:
        var out = x
        out = self.l1.forward(x)
        out = self.l2.forward(out)
        out = self.l3.forward(out)
        out = self.two_stride_pad(out)
        out = self.l4.forward(out)
        out = self.l5.forward(out)
        out = self.l6.forward(out)
        out = self.two_stride_pad(out)
        out = self.l7.forward(out)
        out = self.l8.forward(out)
        out = self.l9.forward(out)
        out = self.two_stride_pad(out)
        out = self.l10.forward(out)
        out = self.l11.forward(out)
        out = self.l12.forward(out)
        out = self.l13.forward(out)
        out = self.l14.forward(out)
        out = self.l15.forward(out)
        out = self.l16.forward(out)
        out = self.l17.forward(out)
        out = self.l18.forward(out)
        out = self.l19.forward(out)
        out = self.metrics_evals(out, noise)

        return out


struct Decoder:
    var l1: Conv2D
    var l2: Conv2D
    var l3: Res_Block
    var l4: Attention_Block
    var l5: Res_Block
    var l6: Res_Block
    var l7: Res_Block
    var l8: Res_Block
    var l9: Upsample
    var l10: Conv2D
    var l11: Res_Block
    var l12: Res_Block
    var l13: Res_Block
    var l14: Upsample
    var l15: Conv2D
    var l16: Res_Block
    var l17: Res_Block
    var l18: Res_Block
    var l19: Upsample
    var l20: Conv2D
    var l21: Res_Block
    var l22: Res_Block
    var l23: Res_Block
    var l24: GroupNorm
    var l25: SiLU
    var l26: Conv2D


    fn __init__(
        inout self,
    ):
        self.l1 = Conv2D(4, 4, kernel_size=1, padding=(0, 0))
        self.l2 = Conv2D(4, 512, kernel_size=3, padding=(1, 1))
        self.l3 = Res_Block(512, 512)
        self.l4 = Attention_Block(512)
        self.l5 = Res_Block(512, 512)
        self.l6 = Res_Block(512, 512)
        self.l7 = Res_Block(512, 512)
        self.l8 = Res_Block(512, 512)
        self.l9 = Upsample(2)
        self.l10 = Conv2D(512, 512, kernel_size=3, padding=(1, 1))
        self.l11 = Res_Block(512, 512)
        self.l12 = Res_Block(512, 512)
        self.l13 = Res_Block(512, 512)
        self.l14 = Upsample(2)
        self.l15 = Conv2D(512, 512, kernel_size=3, padding=(1, 1))
        self.l16 = Res_Block(512, 256)
        self.l17 = Res_Block(256, 256)
        self.l18 = Res_Block(256, 256)
        self.l19 = Upsample(2)
        self.l20 = Conv2D(256, 256, kernel_size=3, padding=(1, 1))
        self.l21 = Res_Block(256, 128)
        self.l22 = Res_Block(128, 128)
        self.l23 = Res_Block(128, 128)
        self.l24 = GroupNorm(32, 128)
        self.l25 = SiLU()
        self.l26 = Conv2D(128, 3, kernel_size=3, padding=(1, 1))

    fn forward(inout self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var out = x / 0.18215
        out = self.l1.forward(out)
        out = self.l2.forward(out)
        out = self.l3.forward(out)
        out = self.l4.forward(out)
        out = self.l5.forward(out)
        out = self.l6.forward(out)
        out = self.l7.forward(out)
        out = self.l8.forward(out)
        out = self.l9.forward(out)
        out = self.l10.forward(out)
        out = self.l11.forward(out)
        out = self.l12.forward(out)
        out = self.l13.forward(out)
        out = self.l14.forward(out)
        out = self.l15.forward(out)
        out = self.l16.forward(out)
        out = self.l17.forward(out)
        out = self.l18.forward(out)
        out = self.l19.forward(out)
        out = self.l20.forward(out)
        out = self.l21.forward(out)
        out = self.l22.forward(out)
        out = self.l23.forward(out)
        out = self.l24.forward(out)
        out = self.l25.forward(out)
        out = self.l26.forward(out)

        return out
