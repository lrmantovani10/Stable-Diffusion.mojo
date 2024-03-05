from .utils import *
import math


struct Self_Attention:
    var n_heads: Int
    var in_proj: Linear
    var out_proj: Linear

    fn __init__(
        inout self,
        n_heads: Int,
        d_embedding: Int,
        in_bias: Bool = True,
        out_bias: Bool = True,
    ):
        self.in_proj = Linear(d_embedding, 3 * d_embedding, in_bias)
        self.out_proj = Linear(d_embedding, d_embedding, out_bias)
        self.n_heads = n_heads

    fn __copyinit__(inout self, other: Self_Attention):
        self.in_proj = other.in_proj
        self.out_proj = other.out_proj
        self.n_heads = other.n_heads

    fn forward(
        inout self, inout x: Matrix[float_dtype], causal_mask: Bool = False
    ) -> Matrix[float_dtype]:
        var chunked_input = self.in_proj.forward(x).chunk(2, 3)
        var q = chunked_input[0].reshape(
            chunked_input[0].dim0 * self.n_heads,
            chunked_input[0].dim1,
            chunked_input[0].dim2 // self.n_heads,
        )
        var k = chunked_input[1].reshape(
            chunked_input[1].dim0 * self.n_heads,
            chunked_input[1].dim1,
            chunked_input[1].dim2 // self.n_heads,
        )
        var v = chunked_input[2].reshape(
            chunked_input[2].dim0 * self.n_heads,
            chunked_input[2].dim1,
            chunked_input[2].dim2 // self.n_heads,
        )

        var weight = q.matmul(k.transpose(1, 2))

        if causal_mask:
            var mask = Matrix[float_dtype](weight.dim0, weight.dim1, weight.dim2)
            mask.set_items(
                slice(0, mask.dim0), slice(0, mask.dim1), slice(0, mask.dim2), 1
            )
            mask = mask.triu(1)
            let neg_inf = math.limit.neginf[float_dtype]()
            weight = weight.masked_fill(mask, neg_inf)

        let head_float: Float32 = x.dim2 // self.n_heads
        weight = weight / math.sqrt(head_float)
        weight = Softmax(weight, dim=2)
        var output = weight.matmul(v)
        output = output.transpose(0, 1)
        output = output.reshape(x.dim0, x.dim1, x.dim2)
        output = self.out_proj.forward(output)

        return output


struct Cross_Attention:
    var n_heads: Int
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
        self.n_heads = n_heads
        self.q_proj = Linear(d_embedding, d_embedding, use_bias=in_bias)
        self.k_proj = Linear(d_crossing, d_embedding, use_bias=in_bias)
        self.v_proj = Linear(d_crossing, d_embedding, use_bias=in_bias)
        self.out_proj = Linear(d_embedding, d_embedding, use_bias=out_bias)

    fn __copyinit__(inout self, other: Cross_Attention):
        self.q_proj = other.q_proj
        self.k_proj = other.k_proj
        self.v_proj = other.v_proj
        self.out_proj = other.out_proj
        self.n_heads = other.n_heads

    fn forward(
        inout self,
        inout x: Matrix[float_dtype],
        inout context: Matrix[float_dtype],
    ) -> Matrix[float_dtype]:
        var q = self.q_proj.forward(x)
        var k = self.k_proj.forward(context)
        var v = self.v_proj.forward(context)

        q = q.reshape(q.dim0 * self.n_heads, q.dim1, q.dim2 // self.n_heads)
        k = k.reshape(k.dim0 * self.n_heads, k.dim1, k.dim2 // self.n_heads)
        v = v.reshape(v.dim0 * self.n_heads, v.dim1, v.dim2 // self.n_heads)

        var weight = q.matmul(k.transpose(1, 2))
        let head_float: Float32 = x.dim2 // self.n_heads
        weight = weight / math.sqrt(head_float)
        weight = Softmax(weight, dim=2)
        var output = weight.matmul(v)
        output = output.transpose(0, 1)
        output = output.reshape(x.dim0, x.dim1, x.dim2)
        output = self.out_proj.forward(output)

        return output
