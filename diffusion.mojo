from helpers.utils import *
from helpers.attention import *


struct Time_Embedding:
    var layer1: Linear
    var layer2: Linear

    fn __init__(inout self, n_embed: Int):
        self.layer1 = Linear(n_embed, 4 * n_embed)
        self.layer2 = Linear(4 * n_embed, 4 * n_embed)

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

    fn forward(
        inout self, x: Matrix[float_dtype], time: Matrix[float_dtype]
    ) -> Matrix[float_dtype]:
        var residue = x
        var out = self.layer1.forward(x)
        out = SiLU().forward(out)
        out = self.layer2.forward(out)
        var time_new = SiLU().forward(time)
        time_new = self.layer3.forward(time_new)
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
        let channels = n_head * n_embed
        self.layer1 = GroupNorm(32, channels)
        self.layer2 = Conv2D(channels, channels, 1, (0, 0))
        self.layer3 = LayerNorm(channels)
        self.layer4 = Self_Attention(n_head, channels, in_bias=False)
        self.layer5 = LayerNorm(channels)
        self.layer6 = Cross_Attention(n_head, channels, d_context, in_bias=False)
        self.layer7 = LayerNorm(channels)
        self.layer8 = Linear(channels, 8 * channels)
        self.layer9 = Linear(4 * channels, channels)
        self.layer10 = Conv2D(channels, channels, 1, (0, 0))

    fn forward(
        inout self, x: Matrix[float_dtype], inout context: Matrix[float_dtype]
    ) -> Matrix[float_dtype]:
        let residue_long = x
        var out = self.layer1.forward(x)
        out = self.layer2.forward(out)
        out = out.reshape(1, out.dim0, out.dim1 * out.dim2)
        out = out.transpose(1, 2)
        var residue_short = out
        out = self.layer3.forward(out)
        out = self.layer4.forward(out)
        out = out + residue_short
        residue_short = out
        out = self.layer5.forward(out)
        out = self.layer6.forward(out, context)
        out = out + residue_short
        residue_short = out
        out = self.layer7.forward(out)

        let chunked_linear = self.layer8.forward(out).chunk(1, 2)
        out = chunked_linear[0]
        var gate = chunked_linear[1]

        out = out.matmul(Gelu().forward(gate))
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
    var layer4: Unet_Residual_Block
    var layer5: Unet_Attention_Block
    var layer6: Conv2D
    var layer7: Unet_Residual_Block
    var layer8: Unet_Attention_Block
    var layer9: Unet_Residual_Block
    var layer10: Unet_Attention_Block
    var layer11: Conv2D
    var layer12: Unet_Residual_Block
    var layer13: Unet_Attention_Block
    var layer14: Unet_Residual_Block
    var layer15: Unet_Attention_Block
    var layer16: Conv2D
    var layer17: Unet_Residual_Block
    var layer18: Unet_Residual_Block
    var layer19: Unet_Residual_Block
    var layer20: Unet_Attention_Block
    var layer21: Unet_Residual_Block
    var layer22: Unet_Residual_Block
    var layer23: Unet_Residual_Block
    var layer24: Unet_Residual_Block
    var layer25: Upsample
    var layer26: Unet_Residual_Block
    var layer27: Unet_Attention_Block
    var layer28: Unet_Residual_Block
    var layer29: Unet_Attention_Block
    var layer30: Unet_Residual_Block
    var layer31: Unet_Attention_Block
    var layer32: Upsample
    var layer33: Unet_Residual_Block
    var layer34: Unet_Attention_Block
    var layer35: Unet_Residual_Block
    var layer36: Unet_Attention_Block
    var layer37: Unet_Residual_Block
    var layer38: Unet_Attention_Block
    var layer39: Upsample
    var layer40: Unet_Residual_Block
    var layer41: Unet_Attention_Block
    var layer42: Unet_Residual_Block
    var layer43: Unet_Attention_Block
    var layer44: Unet_Residual_Block
    var layer45: Unet_Attention_Block
    var skip1: Matrix[float_dtype]
    var skip2: Matrix[float_dtype]
    var skip3: Matrix[float_dtype]
    var skip4: Matrix[float_dtype]
    var skip5: Matrix[float_dtype]
    var skip6: Matrix[float_dtype]
    var skip7: Matrix[float_dtype]
    var skip8: Matrix[float_dtype]
    var skip9: Matrix[float_dtype]
    var skip10: Matrix[float_dtype]
    var skip11: Matrix[float_dtype]
    var skip12: Matrix[float_dtype]

    fn __init__(inout self):
        # Encoders
        self.layer1 = Conv2D(4, 320, 3, (1, 1))
        self.layer2 = Unet_Residual_Block(320, 320)
        self.layer3 = Unet_Attention_Block(8, 40)
        self.layer4 = Unet_Residual_Block(320, 320)
        self.layer5 = Unet_Attention_Block(8, 40)
        self.layer6 = Conv2D(320, 320, 3, (1, 1), stride=(2, 2))
        self.layer7 = Unet_Residual_Block(320, 640)
        self.layer8 = Unet_Attention_Block(8, 80)
        self.layer9 = Unet_Residual_Block(640, 640)
        self.layer10 = Unet_Attention_Block(8, 80)
        self.layer11 = Conv2D(640, 640, 3, (1, 1), stride=(2, 2))
        self.layer12 = Unet_Residual_Block(640, 1280)
        self.layer13 = Unet_Attention_Block(8, 160)
        self.layer14 = Unet_Residual_Block(1280, 1280)
        self.layer15 = Unet_Attention_Block(8, 160)
        self.layer16 = Conv2D(1280, 1280, 3, (1, 1), stride=(2, 2))
        self.layer17 = Unet_Residual_Block(1280, 1280)
        self.layer18 = Unet_Residual_Block(1280, 1280)

        # Bottlenecks
        self.layer19 = Unet_Residual_Block(1280, 1280)
        self.layer20 = Unet_Attention_Block(8, 160)
        self.layer21 = Unet_Residual_Block(1280, 1280)

        # Decoders
        self.layer22 = Unet_Residual_Block(2560, 1280)
        self.layer23 = Unet_Residual_Block(2560, 1280)
        self.layer24 = Unet_Residual_Block(2560, 1280)
        self.layer25 = Upsample(1280)
        self.layer26 = Unet_Residual_Block(2560, 1280)
        self.layer27 = Unet_Attention_Block(8, 160)
        self.layer28 = Unet_Residual_Block(2560, 1280)
        self.layer29 = Unet_Attention_Block(8, 160)
        self.layer30 = Unet_Residual_Block(1920, 640)
        self.layer31 = Unet_Attention_Block(8, 160)
        self.layer32 = Upsample(1280)
        self.layer33 = Unet_Residual_Block(1920, 640)
        self.layer34 = Unet_Attention_Block(8, 80)
        self.layer35 = Unet_Residual_Block(1280, 640)
        self.layer36 = Unet_Attention_Block(8, 80)
        self.layer37 = Unet_Residual_Block(960, 640)
        self.layer38 = Unet_Attention_Block(8, 80)
        self.layer39 = Upsample(640)
        self.layer40 = Unet_Residual_Block(960, 320)
        self.layer41 = Unet_Attention_Block(8, 40)
        self.layer42 = Unet_Residual_Block(640, 320)
        self.layer43 = Unet_Attention_Block(8, 40)
        self.layer44 = Unet_Residual_Block(640, 320)
        self.layer45 = Unet_Attention_Block(8, 40)

        # Skip connections
        self.skip1 = Matrix[float_dtype]()
        self.skip2 = Matrix[float_dtype]()
        self.skip3 = Matrix[float_dtype]()
        self.skip4 = Matrix[float_dtype]()
        self.skip5 = Matrix[float_dtype]()
        self.skip6 = Matrix[float_dtype]()
        self.skip7 = Matrix[float_dtype]()
        self.skip8 = Matrix[float_dtype]()
        self.skip9 = Matrix[float_dtype]()
        self.skip10 = Matrix[float_dtype]()
        self.skip11 = Matrix[float_dtype]()
        self.skip12 = Matrix[float_dtype]()

    fn forward(
        inout self,
        x: Matrix[float_dtype],
        inout context: Matrix[float_dtype],
        inout time: Matrix[float_dtype],
    ) -> Matrix[float_dtype]:
        # Encoders
        var out = self.layer1.forward(x)
        self.skip1 = out
        out = self.layer2.forward(out, time)
        out = self.layer3.forward(out, context)
        self.skip2 = out
        out = self.layer4.forward(out, time)
        out = self.layer5.forward(out, context)
        self.skip3 = out
        out = self.layer6.forward(out)
        self.skip4 = out
        out = self.layer7.forward(out, time)
        out = self.layer8.forward(out, context)
        self.skip5 = out
        out = self.layer9.forward(out, time)
        out = self.layer10.forward(out, context)
        self.skip6 = out
        out = self.layer11.forward(out)
        self.skip7 = out
        out = self.layer12.forward(out, time)
        out = self.layer13.forward(out, context)
        self.skip8 = out
        out = self.layer14.forward(out, time)
        out = self.layer15.forward(out, context)
        self.skip9 = out
        out = self.layer16.forward(out)
        self.skip10 = out
        out = self.layer17.forward(out, time)
        self.skip11 = out
        out = self.layer18.forward(out, time)
        self.skip12 = out

        # Bottlenecks
        out = self.layer19.forward(out, time)
        out = self.layer20.forward(out, context)
        out = self.layer21.forward(out, time)

        # Decoders
        out = out.concat(self.skip1, 1)
        out = self.layer22.forward(out, time)
        out = out.concat(self.skip2, 1)
        out = self.layer23.forward(out, time)
        out = out.concat(self.skip3, 1)
        out = self.layer24.forward(out, time)
        out = self.layer25.forward(out)
        out = out.concat(self.skip4, 1)
        out = self.layer26.forward(out, time)
        out = self.layer27.forward(out, context)
        out = out.concat(self.skip5, 1)
        out = self.layer28.forward(out, time)
        out = self.layer29.forward(out, context)
        out = out.concat(self.skip6, 1)
        out = self.layer30.forward(out, time)
        out = self.layer31.forward(out, context)
        out = self.layer32.forward(out)
        out = out.concat(self.skip7, 1)
        out = self.layer33.forward(out, time)
        out = self.layer34.forward(out, context)
        out = out.concat(self.skip8, 1)
        out = self.layer35.forward(out, time)
        out = self.layer36.forward(out, context)
        out = out.concat(self.skip9, 1)
        out = self.layer37.forward(out, time)
        out = self.layer38.forward(out, context)
        out = self.layer39.forward(out)
        out = out.concat(self.skip10, 1)
        out = self.layer40.forward(out, time)
        out = self.layer41.forward(out, context)
        out = out.concat(self.skip11, 1)
        out = self.layer42.forward(out, time)
        out = self.layer43.forward(out, context)
        out = out.concat(self.skip12, 1)
        out = self.layer44.forward(out, time)
        out = self.layer45.forward(out, context)

        return out


struct UNet_Output_Layer:
    var layer1: GroupNorm
    var layer2: Conv2D

    fn __init__(inout self, in_channels: Int, out_channels: Int):
        self.layer1 = GroupNorm(32, in_channels)
        self.layer2 = Conv2D(in_channels, out_channels, 3, (1, 1))

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
