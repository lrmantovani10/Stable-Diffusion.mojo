from .utils import *
import math


struct Self_Attention:
    var n_heads: Int
    var d_head: Int
    var in_proj: Linear
    var out_proj: Linear

    fn __init__(
        inout self,
        n_heads: Int,
        d_embedding: Int,
        num_channels: Int,
        in_bias: Bool = True,
        out_bias: Bool = True,
    ):
        self.in_proj = Linear(d_embedding, 3 * d_embedding, num_channels, in_bias)
        self.out_proj = Linear(d_embedding, d_embedding, num_channels, out_bias)
        self.n_heads = n_heads
        self.d_head = d_embedding // self.n_heads

    fn forward(
        inout self, inout x: Matrix[float_dtype], causal_mask: Bool = False
    ) -> Matrix[float_dtype]:
        var chunked_input = self.in_proj.forward(x).chunk(2, 3)

        var q = chunked_input[0].reshape(x.dim0, self.n_heads, self.d_head)
        q = q.transpose(1, 2)
        var k = chunked_input[1].reshape(x.dim0, self.n_heads, self.d_head)
        k = k.transpose(1, 2)
        var v = chunked_input[2].reshape(x.dim0, self.n_heads, self.d_head)
        v = v.transpose(1, 2)

        var weight = q.matmul(k.transpose(k.dim1, k.dim2))

        if causal_mask:
            var mask = Matrix[float_dtype](weight.dim0, weight.dim1, weight.dim2)
            mask.set_items(
                slice(0, mask.dim0), slice(0, mask.dim1), slice(0, mask.dim2), 1
            )
            mask = mask.triu(1)
            let neg_inf = math.limit.neginf[float_dtype]()
            weight = weight.masked_fill(mask, neg_inf)

        weight = weight / math.sqrt(self.d_head)
        weight = Softmax(weight, dim=2)
        var output = weight.matmul(v)
        output = output.transpose(1, 2)
        output = output.reshape(x.dim0, x.dim1, x.dim2)
        output = self.out_proj.forward(output)

        return output


struct CrossAttention:
    var n_heads: Int
    var d_head: Int
    var q_proj: Linear
    var k_proj: Linear
    var v_proj: Linear
    var out_proj: Linear

    fn __init__(
        inout self,
        n_heads: Int,
        d_embedding: Int,
        d_crossing: Int,
        in_bias: Bool = True,
        out_bias: Bool = True,
    ):
        self.q_proj = Linear(d_embedding, d_embedding, use_bias=in_bias)
        self.k_proj = Linear(d_crossing, d_embedding, use_bias=in_bias)
        self.v_proj = Linear(d_crossing, d_embedding, use_bias=in_bias)
        self.out_proj = Linear(d_embedding, d_embedding, use_bias=out_bias)
        self.n_heads = n_heads
        self.d_head = d_embedding // n_heads

    fn forward(
        inout self,
        inout x: Matrix[float_dtype],
        inout context: Matrix[float_dtype],
    ) -> Matrix[float_dtype]:
        var q = self.q_proj.forward(x)
        var k = self.k_proj.forward(context)
        var v = self.v_proj.forward(context)

        q = q.reshape(
            int(q.size() / (self.n_heads * self.d_head)), self.n_heads, self.d_head
        )
        q = q.transpose(0, 1)
        k = k.reshape(
            int(k.size() / (self.n_heads * self.d_head)), self.n_heads, self.d_head
        )
        k = k.transpose(0, 1)
        v = v.reshape(
            int(v.size() / (self.n_heads * self.d_head)), self.n_heads, self.d_head
        )
        v = v.transpose(0, 1)

        var weight = q.matmul(k.transpose(1, 2))

        weight = weight / math.sqrt(self.d_head)
        weight = Softmax(weight, dim=2)
        var output = weight.matmul(v)
        output = output.transpose(0, 1)
        output = output.reshape(x.dim0, x.dim1, x.dim2)
        output = self.out_proj.forward(output)

        return output
