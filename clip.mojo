from helpers.utils import *
from helpers.attention import *


struct ClipEmbedding:
    var token_embedding: Embedding

    ## LEARNABLE PARAMETER
    var position_embedding: Matrix[float_dtype]

    fn __init__(inout self, n_vocab: Int, n_embed: Int, n_token: Int):
        self.token_embedding = Embedding(n_vocab, n_embed)
        let pos_embed_matrix = Matrix[float_dtype](1, n_token, n_embed)
        pos_embed_matrix *= 0
        self.position_embedding = pos_embed_matrix

    fn forward(self, tokens: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var out = self.token_embedding.forward(tokens)
        out = out + self.position_embedding
        return out


struct ClipPlayer:
    var layer1: LayerNorm
    var layer2: Self_Attention
    var layer3: LayerNorm
    var layer4: Linear
    var layer5: Linear

    fn __init__(inout self, n_head: Int, n_embed: Int):
        self.layer1 = LayerNorm(n_embed)
        self.layer2 = Self_Attention(n_head, n_embed)
        self.layer3 = LayerNorm(n_embed)
        self.layer4 = Linear(n_embed, 4 * n_embed)
        self.layer5 = Linear(4 * n_embed, n_embed)

    fn forward(inout self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var residue = x
        var first_input = residue.transpose(0, 2)
        var out = self.layer1.forward(first_input)
        out = out.transpose(0, 2)
        out = self.layer2.forward(out, causal_mask=True)
        out = out + residue
        residue = out
        out = out.transpose(0, 2)
        out = self.layer3.forward(out)
        out = out.transpose(0, 2)
        out = self.layer4.forward(out)
        var out_multiplied = out * 1.702
        out = out.multiply(sigmoid(out_multiplied))
        out = self.layer5.forward(out)
        out = out + residue
        return out


struct CLIP:
    var embedding: ClipEmbedding
    var player1: ClipPlayer
    var player2: ClipPlayer
    var player3: ClipPlayer
    var player4: ClipPlayer
    var player5: ClipPlayer
    var player6: ClipPlayer
    var player7: ClipPlayer
    var player8: ClipPlayer
    var player9: ClipPlayer
    var player10: ClipPlayer
    var player11: ClipPlayer
    var player12: ClipPlayer
    var layer_norm: LayerNorm

    fn __init__(inout self):
        self.embedding = ClipEmbedding(49408, 768, 77)
        self.player1 = ClipPlayer(12, 768)
        self.player2 = ClipPlayer(12, 768)
        self.player3 = ClipPlayer(12, 768)
        self.player4 = ClipPlayer(12, 768)
        self.player5 = ClipPlayer(12, 768)
        self.player6 = ClipPlayer(12, 768)
        self.player7 = ClipPlayer(12, 768)
        self.player8 = ClipPlayer(12, 768)
        self.player9 = ClipPlayer(12, 768)
        self.player10 = ClipPlayer(12, 768)
        self.player11 = ClipPlayer(12, 768)
        self.player12 = ClipPlayer(12, 768)
        self.layer_norm = LayerNorm(768)

    fn forward(inout self, inout tokens: Matrix[float_dtype]) -> Matrix[float_dtype]:
        # Here, we do not convert "state" to the long type (float64)for simplicity in Mojo type handling, but in production it would be useful to copy and paste the body of all these functions with type float64 instead of float32
        var reshaped_tokens = Matrix[float_dtype](1, 1, 77)
        reshaped_tokens *= 0
        reshaped_tokens.set_items(0, 0, slice(0, tokens.dim2), tokens)
        var state = self.embedding.forward(reshaped_tokens)
        state = self.player1.forward(state)
        state = self.player2.forward(state)
        state = self.player3.forward(state)
        state = self.player4.forward(state)
        state = self.player5.forward(state)
        state = self.player6.forward(state)
        state = self.player7.forward(state)
        state = self.player8.forward(state)
        state = self.player9.forward(state)
        state = self.player10.forward(state)
        state = self.player11.forward(state)
        state = self.player12.forward(state)
        state = state.transpose(0, 2)
        var output = self.layer_norm.forward(state)
        output = output.transpose(0, 2)
        return output
