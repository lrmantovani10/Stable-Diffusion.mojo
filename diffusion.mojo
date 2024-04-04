from helpers.utils import *
from helpers.attention import *


struct Time_Embedding:
    var layer1: Linear
    var layer2: Linear

    fn __init__(inout self, n_embed: Int):
        self.layer1 = Linear(n_embed, 4 * n_embed)
        self.layer2 = Linear(4 * n_embed, 4 * n_embed)

    fn __copyinit__(inout self, other: Self):
        self.layer1 = other.layer1
        self.layer2 = other.layer2

    fn forward(inout self, inout x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var out = self.layer1.forward(x)
        out = SiLU().forward(out)
        out = self.layer2.forward(out)
        return out


struct Unet_Residual_Block:
    var layer1: GroupNorm
    var layer2: Conv2D
    var layer3: Linear
    var layer4: GroupNorm
    var layer5: Conv2D
    var layer6: Conv2D
    var in_channels: Int
    var out_channels: Int

    fn __init__(inout self, in_channels: Int, out_channels: Int, n_time: Int = 1280):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer1 = GroupNorm(32, in_channels)
        self.layer2 = Conv2D(in_channels, out_channels, kernel_size=3, padding=(1, 1))
        self.layer3 = Linear(n_time, out_channels)
        self.layer4 = GroupNorm(32, out_channels)
        self.layer5 = Conv2D(out_channels, out_channels, 3, (1, 1))
        self.layer6 = Conv2D(in_channels, out_channels, 1, (0, 0))

    fn __copyinit__(inout self, other: Self):
        self.layer1 = other.layer1
        self.layer2 = other.layer2
        self.layer3 = other.layer3
        self.layer4 = other.layer4
        self.layer5 = other.layer5
        self.layer6 = other.layer6
        self.in_channels = other.in_channels
        self.out_channels = other.out_channels

    fn forward(
        inout self, x: Matrix[float_dtype], time: Matrix[float_dtype]
    ) -> Matrix[float_dtype]:
        var residue = x
        var out = self.layer1.forward(x)
        out = SiLU().forward(out)
        out = self.layer2.forward(out)
        var time_new = SiLU().forward(time)
        time_new = self.layer3.forward(time_new)
        time_new = time_new.reshape(self.out_channels, 1, 1)
        time_new = time_new.broadcast_channel(out.dim1, out.dim2)
        var merged = out + time_new
        merged = self.layer4.forward(merged)
        merged = SiLU().forward(merged)

        merged = self.layer5.forward(merged)
        if self.in_channels != self.out_channels:
            return merged + self.layer6.forward(residue)
        return merged + residue


struct Unet_Attention_Block:
    var layer1: GroupNorm
    var layer2: Conv2D
    var layer3: LayerNorm
    var layer4: Self_Attention
    var layer5: LayerNorm
    var layer6: Cross_Attention
    var layer7: LayerNorm
    var layer8: Linear
    var layer9: Linear
    var layer10: Conv2D

    fn __init__(inout self, n_head: Int, n_embed: Int, d_context: Int = 768):
        var channels = n_head * n_embed
        self.layer1 = GroupNorm(32, channels, epsilon=1e-6)
        self.layer2 = Conv2D(channels, channels, 1, (0, 0))
        self.layer3 = LayerNorm(channels)
        self.layer4 = Self_Attention(n_head, channels, in_bias=False)
        self.layer5 = LayerNorm(channels)
        self.layer6 = Cross_Attention(n_head, channels, d_context, in_bias=False)
        self.layer7 = LayerNorm(channels)
        self.layer8 = Linear(channels, 8 * channels)
        self.layer9 = Linear(4 * channels, channels)
        self.layer10 = Conv2D(channels, channels, 1, (0, 0))

    fn __copyinit__(inout self, other: Self):
        self.layer1 = other.layer1
        self.layer2 = other.layer2
        self.layer3 = other.layer3
        self.layer4 = other.layer4
        self.layer5 = other.layer5
        self.layer6 = other.layer6
        self.layer7 = other.layer7
        self.layer8 = other.layer8
        self.layer9 = other.layer9
        self.layer10 = other.layer10

    fn forward(
        inout self, x: Matrix[float_dtype], inout context: Matrix[float_dtype]
    ) -> Matrix[float_dtype]:
        var residue_long = x
        var out = self.layer1.forward(x)
        out = self.layer2.forward(out)
        out = out.reshape(1, out.dim0, out.dim1 * out.dim2)
        out = out.transpose(1, 2)
        out = out.transpose(0, 2)
        var residue_short = out
        out = self.layer3.forward(out)
        out = out.transpose(0, 2)
        out = self.layer4.forward(out)
        residue_short = residue_short.transpose(0, 2)
        out = out + residue_short
        residue_short = out
        out = out.transpose(0, 2)
        out = self.layer5.forward(out)
        out = out.transpose(0, 2)

        out = self.layer6.forward(out, context)
        out = out + residue_short
        residue_short = out
        out = out.transpose(0, 2)
        out = self.layer7.forward(out)
        out = out.transpose(0, 2)
        var chunked_linear = self.layer8.forward(out).chunk(2, 2)
        out = chunked_linear[0]
        var gate = chunked_linear[1]
        out = out.multiply(Gelu().forward(gate))
        out = self.layer9.forward(out)
        out = out + residue_short
        out = out.transpose(1, 2)
        out = out.reshape(x.dim0, x.dim1, x.dim2)
        out = self.layer10.forward(out) + residue_long
        return out


struct UNet:
    var layer1: Conv2D
    var layer2: Unet_Residual_Block
    var layer3: Unet_Attention_Block
    var layer4: Conv2D
    var layer5: Unet_Residual_Block
    var layer6: Unet_Attention_Block
    var layer7: Conv2D
    var layer8: Unet_Residual_Block
    var layer9: Unet_Attention_Block
    var layer10: Unet_Residual_Block
    var layer11: Unet_Attention_Block
    var layer12: Unet_Residual_Block
    var layer13: Unet_Attention_Block
    var layer14: Upsample
    var layer15: Unet_Residual_Block
    var layer16: Unet_Attention_Block
    var layer17: Unet_Residual_Block
    var layer18: Unet_Attention_Block
    var layer19: Upsample
    var layer20: Unet_Residual_Block
    var layer21: Unet_Attention_Block
    var layer22: Unet_Residual_Block
    var layer23: Unet_Attention_Block

    fn __init__(inout self):
        # Encoders
        self.layer1 = Conv2D(4, 320, 3, (1, 1))
        self.layer2 = Unet_Residual_Block(320, 320)
        self.layer3 = Unet_Attention_Block(8, 40)
        self.layer4 = Conv2D(320, 320, 3, (1, 1), (2, 2))
        self.layer5 = Unet_Residual_Block(320, 640)
        self.layer6 = Unet_Attention_Block(8, 80)
        self.layer7 = Conv2D(640, 640, 3, (1, 1), (2, 2))
        self.layer8 = Unet_Residual_Block(640, 1280)
        self.layer9 = Unet_Attention_Block(8, 160)

        # Decoders
        self.layer10 = Unet_Residual_Block(2560, 1280)
        self.layer11 = Unet_Attention_Block(8, 160)
        self.layer12 = Unet_Residual_Block(1920, 1280)
        self.layer13 = Unet_Attention_Block(8, 160)
        self.layer14 = Upsample(1280)
        self.layer15 = Unet_Residual_Block(1280, 640)
        self.layer16 = Unet_Attention_Block(8, 80)
        self.layer17 = Unet_Residual_Block(960, 640)
        self.layer18 = Unet_Attention_Block(8, 80)
        self.layer19 = Upsample(640)
        self.layer20 = Unet_Residual_Block(640, 320)
        self.layer21 = Unet_Attention_Block(8, 40)
        self.layer22 = Unet_Residual_Block(640, 320)
        self.layer23 = Unet_Attention_Block(8, 40)

    fn __copyinit__(inout self, other: Self):
        self.layer1 = other.layer1
        self.layer2 = other.layer2
        self.layer3 = other.layer3
        self.layer4 = other.layer4
        self.layer5 = other.layer5
        self.layer6 = other.layer6
        self.layer7 = other.layer7
        self.layer8 = other.layer8
        self.layer9 = other.layer9
        self.layer10 = other.layer10
        self.layer11 = other.layer11
        self.layer12 = other.layer12
        self.layer13 = other.layer13
        self.layer14 = other.layer14
        self.layer15 = other.layer15
        self.layer16 = other.layer16
        self.layer17 = other.layer17
        self.layer18 = other.layer18
        self.layer19 = other.layer19
        self.layer20 = other.layer20
        self.layer21 = other.layer21
        self.layer22 = other.layer22
        self.layer23 = other.layer23

    fn forward(
        inout self,
        x: Matrix[float_dtype],
        inout context: Matrix[float_dtype],
        inout time: Matrix[float_dtype],
    ) -> Matrix[float_dtype]:
        
        # Encoders
        var out = self.layer1.forward(x)
        var skip1 = out
        out = self.layer2.forward(out, time)
        out = self.layer3.forward(out, context)
        var skip2 = out
        out = self.layer4.forward(out)
        var skip3 = out
        out = self.layer5.forward(out, time)
        out = self.layer6.forward(out, context)
        var skip4 = out
        out = self.layer7.forward(out)
        var skip5 = out
        out = self.layer8.forward(out, time)
        out = self.layer9.forward(out, context)
        var skip6 = out

        # Decoders
        out = out.concat(skip6, 0)
        out = self.layer10.forward(out, time)
        out = self.layer11.forward(out, context)
        out = out.concat(skip5, 0)
        out = self.layer12.forward(out, time)
        out = self.layer13.forward(out, context)
        out = self.layer14.forward(out)
        out = out.concat(skip4, 0)
        out = self.layer15.forward(out, time)
        out = self.layer16.forward(out, context)
        out = out.concat(skip3, 0)
        out = self.layer17.forward(out, time)
        out = self.layer18.forward(out, context)
        out = self.layer19.forward(out)
        out = out.concat(skip2, 0)
        out = self.layer20.forward(out, time)
        out = self.layer21.forward(out, context)
        out = out.concat(skip1, 0)
        out = self.layer22.forward(out, time)
        out = self.layer23.forward(out, context)
        return out

struct UNet_Output_Layer:
    var layer1: GroupNorm
    var layer2: Conv2D

    fn __init__(inout self, in_channels: Int, out_channels: Int):
        self.layer1 = GroupNorm(320, in_channels)
        self.layer2 = Conv2D(in_channels, out_channels, 3, (1, 1))

    fn __copyinit__(inout self, other: Self):
        self.layer1 = other.layer1
        self.layer2 = other.layer2

    fn forward(inout self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var out = self.layer1.forward(x)
        out = SiLU().forward(out)
        out = self.layer2.forward(out)
        return out


struct Diffusion:
    var time_embed: Time_Embedding
    var unet: UNet
    var final: UNet_Output_Layer

    fn __init__(inout self):
        self.time_embed = Time_Embedding(320)
        self.unet = UNet()   
        self.final = UNet_Output_Layer(320, 4)

    fn __copyinit__(inout self, other: Self):
        self.time_embed = other.time_embed
        self.unet = other.unet
        self.final = other.final

    fn forward(
        inout self,
        x: Matrix[float_dtype],
        inout context: Matrix[float_dtype],
        inout time: Matrix[float_dtype],
    ) -> Matrix[float_dtype]:
        var time_embedded = self.time_embed.forward(time)
        var out = self.unet.forward(x, context, time_embedded)
        out = self.final.forward(out)
        return out
