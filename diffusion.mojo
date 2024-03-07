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

    fn __init__(inout self, in_channels: Int, out_channels: Int, n_time: Int = 128):
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
        out = out.transpose(1, 2)
        out = out.matmul(Gelu().forward(gate))
        out = self.layer9.forward(out)

        ## In these lines, I am concatenating the "out" variable multiple times because the latent space dimension (128 rows here) is much smaller than the "residue short" variable, which was constructed from a real Stable Diffusion tokenizer (which is why it has 4096 rows).
        # However, this should not be done in production. The correct approach is either to use a smaller tokenizer or a larger latent space.
        var original_out = out
        var diff = residue_short.dim1 // out.dim1 - 1
        if diff > 0:
            for _ in range(residue_short.dim1 // out.dim1 - 1):
                out = out.concat(original_out, 1)

        out = out + residue_short
        out = out.transpose(1, 2)
        out = out.reshape(x.dim0, x.dim1, x.dim2)
        out = self.layer10.forward(out) + residue_long
        return out


# A much smaller UNet compared to the original Stable Diffusion's
struct UNet:
    var layer1: Conv2D
    var layer2: Unet_Residual_Block
    var layer3: Unet_Attention_Block
    var layer4: Unet_Residual_Block
    var layer5: Unet_Attention_Block
    var layer6: Conv2D
    var layer7: Unet_Residual_Block
    var layer8: Unet_Attention_Block
    var layer9: Unet_Residual_Block
    var layer10: Unet_Attention_Block
    var layer11: Unet_Residual_Block
    var layer12: Unet_Attention_Block
    var layer13: Upsample
    var layer14: Unet_Residual_Block
    var layer15: Unet_Attention_Block
    var layer16: Unet_Residual_Block
    var layer17: Unet_Attention_Block

    fn __init__(inout self):
        # Encoders
        self.layer1 = Conv2D(4, 32, 3, (1, 1))
        self.layer2 = Unet_Residual_Block(32, 32)
        self.layer3 = Unet_Attention_Block(8, 4)
        self.layer4 = Unet_Residual_Block(32, 32)
        self.layer5 = Unet_Attention_Block(8, 4)
        self.layer6 = Conv2D(32, 32, 3, (1, 1))

        # Bottleneck
        self.layer7 = Unet_Residual_Block(32, 64)
        self.layer8 = Unet_Attention_Block(8, 8)

        # Decoders
        self.layer9 = Unet_Residual_Block(96, 64)
        self.layer10 = Unet_Attention_Block(8, 8)

        # Here I use a very small upsampling factor for demonstrative purposes
        self.layer11 = Unet_Residual_Block(96, 32)
        self.layer12 = Unet_Attention_Block(8, 4)
        self.layer13 = Upsample(2)
        self.layer14 = Unet_Residual_Block(96, 32)
        self.layer15 = Unet_Attention_Block(8, 4)
        self.layer16 = Unet_Residual_Block(64, 32)
        self.layer17 = Unet_Attention_Block(8, 4)

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
        out = self.layer4.forward(out, time)
        out = self.layer5.forward(out, context)
        var skip3 = out
        out = self.layer6.forward(out)
        var skip4 = out

        # Bottleneck
        out = self.layer7.forward(out, time)
        out = self.layer8.forward(out, context)

        # Decoders
        out = out.concat(skip1, 0)
        out = self.layer9.forward(out, time)
        out = self.layer10.forward(out, context)
        out = self.layer11.forward(out, time)
        out = self.layer12.forward(out, context)
        out = self.layer13.forward(out)
        out = out.concat(skip2, 0)
        out = out.concat(skip3, 0)
        out = self.layer14.forward(out, time)
        out = self.layer15.forward(out, context)
        out = out.concat(skip4, 0)
        out = self.layer16.forward(out, time)
        out = self.layer17.forward(out, context)

        return out


struct UNet_Output_Layer:
    var layer1: GroupNorm
    var layer2: Conv2D

    fn __init__(inout self, in_channels: Int, out_channels: Int):
        self.layer1 = GroupNorm(32, in_channels)
        self.layer2 = Conv2D(in_channels, out_channels, 3, (1, 1))

    fn __copyinit__(inout self, other: Self):
        self.layer1 = other.layer1
        self.layer2 = other.layer2

    fn forward(inout self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var out = self.layer1.forward(x)
        out = SiLU().forward(out)
        out = self.layer2.forward(out)
        return out


# Here, I reduced the values used to initialize Time_Embedding and UNet_Output_Layer from 320 to 32 for faster testing time.
struct Diffusion:
    var time_embed: Time_Embedding
    var unet: UNet
    var final: UNet_Output_Layer

    fn __init__(inout self):
        self.time_embed = Time_Embedding(32)
        self.unet = UNet()
        self.final = UNet_Output_Layer(32, 4)

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
