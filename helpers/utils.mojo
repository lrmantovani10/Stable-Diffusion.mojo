from tensor import Tensor, TensorShape
from algorithm import parallelize, vectorize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from random import rand, random_float64, randn_float64
from sys.info import simdwidthof
from memory import memset_zero, memcpy
from sys.intrinsics import strided_load
from math import trunc, mod, cos, sin, round
from random import random_ui64, seed
import buffer

alias float_base = Float32
alias float_dtype = DType.float32
alias tensor_type = Tensor[float_dtype]
alias simd_width: Int = simdwidthof[float_dtype]()
alias pi = 3.141592653589793238462643383279

fn linspace(start: Float32, end: Float32, steps: Int) -> Tensor[float_dtype]:
    var step = (end - start) / (steps - 1)
    var out = Tensor[float_dtype](steps)
    for i in range(steps):
        out[i] = start + step * i
    return out

fn arange(start: Float32, end: Float32, reverse:Bool = False) -> Tensor[float_dtype]:
    var out: Tensor[float_dtype]
    if not reverse:
        out = Tensor[float_dtype](int(end - start))
        for i in range(end - start):
            out[i] = start + i
    else:
        out = Tensor[float_dtype](int(end - start))
        for i in range(end - start):
            out[i] = end - 1 - i
    return out

# Cumprod for a 1d tensor
fn cumprod(tensor: Tensor[float_dtype]) -> Tensor[float_dtype]:
    var out = Tensor[float_dtype](tensor.num_elements())
    var acc:Float32 = 1
    for i in range(tensor.num_elements()):
        acc *= tensor[i]
        out[i] = acc
    return out

fn round_tensor(tensor: Tensor[float_dtype]) -> Tensor[float_dtype]:
    var out = Tensor[float_dtype](tensor.shape())
    @parameter
    fn round_fn[width: Int](index: Int):
        var val = tensor.load[width=width](index)
        var val_round = round[float_dtype, width](val)
        out.store[width](index, val_round)
    vectorize[round_fn, simd_width](tensor.num_elements())
    return out

fn get_tensor_values(tensor: Tensor[float_dtype], start_index: Int, end_index: Int) -> Tensor[float_dtype]:
    var out = Tensor[float_dtype](end_index - start_index)
    for i in range(start_index, end_index):
        out[i - start_index] = tensor[i]
    return out

# This tokenizer-related section of the code was copied and then modified from the wonderful Mojo Llama2 project available here - https://github.com/tairov/llama2.mojo/blob/master/
struct FileBuf:
    var data: DTypePointer[DType.uint8]
    var offset: Int
    var size: Int

    fn __init__(inout self):
        self.data = DTypePointer[DType.uint8]()
        self.offset = 0
        self.size = 0

    fn __del__(owned self):
        self.data.free()

    fn move_offset(inout self, size: Int) raises:
        var new_offset = self.offset + size
        if new_offset > self.size:
            raise Error("Resulting offset will be past the end of the FileBuf")
        if new_offset < 0:
            raise Error("Resulting offset will be before the beginning of the FileBuf")
        self.offset = new_offset

    fn bitcast_offset_f32(inout self, size: Int) -> DTypePointer[DType.float32]:
        try:
            var ret = self.data.offset(self.offset).bitcast[DType.float32]()
            self.move_offset(size * sizeof[DType.float32]())
            return ret
        except:
            print("Error offsetting float32 while reading float from tokenizer file")
            return DTypePointer[DType.float32]()

    fn get_offset(self) raises -> Int:
        if self.offset > self.size:
            raise Error("Offset is past the end of the FileBuf")
        if self.offset < 0:
            raise Error("Offset is before the beginning of the FileBuf")
        return self.offset

fn read_file(file_name: String, inout buf: FileBuf):
    try:
        var fd = open(file_name, "r")
        var data = fd.read()
        fd.close()
        buf.size = data._buffer.size
        buf.data = data._steal_ptr().bitcast[DType.uint8]()
        buf.offset = 0
    except:
        print("Error reading file")
    return

fn read_val_int(inout buf: FileBuf) -> Int:
    try:
        var data = buf.data.offset(buf.get_offset()).bitcast[DType.int32]()
        var result = data.load(0)
        buf.move_offset(4)
        return result.to_int()
    except:
        print("Error reading int from tokenizer file")
        return 0

fn read_val_float32(inout buf: FileBuf) -> Float32:
    try:
        var val = buf.data.offset(buf.get_offset()).bitcast[DType.float32]().load(0)
        buf.move_offset(4)
        return val
    except:
        print("Error reading float32 from tokenizer file")
        return 0

fn read_val_str(inout buf: FileBuf, slen: Int) -> Pointer[UInt8]:
    try:
        var str = Pointer[UInt8].alloc(slen + 1)
        for i in range(slen):
            str.store(i, buf.data.load(buf.get_offset()))
            buf.move_offset(1)
        str.store(slen, 0)
        return str
    except:
        print("Error reading string from tokenizer file")
        return Pointer[UInt8].alloc(slen + 1)

fn string_compare(a: Pointer[UInt8], b: Pointer[UInt8]) -> Int:
    var index = 0
    while a[index] != 0 and b[index] != 0:
        if a[index] < b[index]:
            return -1
        if a[index] > b[index]:
            return 1

        index += 1

    if a[index] != 0 and b[index] == 0:
        return 1

    if a[index] == 0 and b[index] != 0:
        return -1
    return 0

fn partition(
    inout array: Pointer[Pointer[UInt8]], inout indices: List[Int], low: Int, high: Int
) -> Int:
    var pivot = array[high]
    var ii = low - 1
    for jj in range(low, high):
        if string_compare(pivot, array[jj]) == 1:
            ii = ii + 1

            var tmp = array[ii]
            var tmp_idx = indices[ii]
            array.store(ii, array[jj])
            indices[ii] = indices[jj]
            array.store(jj, tmp)
            indices[jj] = tmp_idx

    var tmp = array[ii + 1]
    var tmp_idx = indices[ii + 1]
    array.store(ii + 1, array[high])
    indices[ii + 1] = indices[high]
    array.store(high, tmp)
    indices[high] = tmp_idx

    return ii + 1

fn quicksort(
    inout array: Pointer[Pointer[UInt8]], inout indices: List[Int], low: Int, high: Int
):
    if low < high:
        var pi = partition(array, indices, low, high)
        quicksort(array, indices, low, pi - 1)
        quicksort(array, indices, pi + 1, high)

fn str_to_ptr(s: String) -> Pointer[UInt8]:
    var ret = Pointer[UInt8].alloc(len(s) + 1)
    for i in range(len(s)):
        ret.store(i, ord(s[i]))
    ret.store(len(s), 0)
    return ret

fn wrap(token: Pointer[UInt8]) -> Pointer[UInt8]:
    if string_compare(token, str_to_ptr("\\n")) == 0:
        return str_to_ptr("<0x0A>")
    if string_compare(token, str_to_ptr("\\t")) == 0:
        return str_to_ptr("<0x09>")
    if string_compare(token, str_to_ptr("'")) == 0:
        return str_to_ptr("<0x27>")
    elif string_compare(token, str_to_ptr('"')) == 0:
        return str_to_ptr("<0x22>")
    return token

fn str_len(s: Pointer[UInt8]) -> Int:
    var len = 0
    while s[len] != 0:
        len += 1
    return len

fn str_concat(s1: Pointer[UInt8], s2: Pointer[UInt8]) -> Pointer[UInt8]:
    var l1 = str_len(s1)
    var l2 = str_len(s2)
    var string = Pointer[UInt8].alloc(l1 + l2 + 1)
    for i in range(l1):
        memcpy[count=1](string.offset(i), s1)
    for i in range(l2):
        memcpy[count=1](string.offset(l1 + i), s2)
    string.store(l1 + l2, 0)
    return string

struct Tokenizer:
    var vocab: Pointer[Pointer[UInt8]]
    var vocab_scores: DTypePointer[DType.float32]
    var max_token_length: Int
    var vocab_size: Int
    var sorted_vocab: Pointer[Pointer[UInt8]]
    var sorted_indices: List[Int]

    fn __init__(inout self, vocab_size: Int, inout buf: FileBuf) -> None:
        self.vocab_size = vocab_size
        self.max_token_length = read_val_int(buf)
        self.vocab_scores = DTypePointer[DType.float32].alloc(self.vocab_size)
        self.vocab = Pointer[Pointer[UInt8]].alloc(self.vocab_size)
        self.sorted_vocab = Pointer[Pointer[UInt8]].alloc(0)
        self.sorted_indices = List[Int](capacity=0)

        for i in range(0, self.vocab_size):
            var score = read_val_float32(buf)
            var slen = read_val_int(buf)
            var token = read_val_str(buf, slen)
            self.store_token(i, token, score)
        return None

    fn __del__(owned self):
        for i in range(0, self.vocab_size):
            self.vocab[i].free()
        self.vocab.free()
        self.vocab_scores.free()
        self.sorted_vocab.free()

    fn store_token(
        inout self, index: Int, owned token: Pointer[UInt8], score: Float32
    ) -> None:
        self.vocab_scores.store(index, score)
        self.vocab.store(index, token)

    fn sort(inout self) -> None:
        if len(self.sorted_indices) < self.vocab_size:
            self.sorted_indices = List[Int](capacity=self.vocab_size)
            self.sorted_vocab = Pointer[Pointer[UInt8]].alloc(self.vocab_size)
            for ii in range(self.vocab_size):
                self.sorted_vocab.store(ii, self.vocab[ii])
                self.sorted_indices.append(ii)

        var n = self.vocab_size
        quicksort(self.sorted_vocab, self.sorted_indices, 0, n - 1)
        return None

    fn find(inout self, token_o: Pointer[UInt8]) -> Int:
        var token = wrap(token_o)
        var n = self.vocab_size
        if len(self.sorted_indices) < n:
            self.sort()
        var left = 0
        var right = n - 1
        while left <= right:
            var mid = left + (right - left) // 2
            var comparison = string_compare(self.sorted_vocab[mid], token)
            if comparison == 0:
                return self.sorted_indices[mid]
            if comparison < 0:
                left = mid + 1
            else:
                right = mid - 1
        return -1

fn bpe_encode(text: String, inout tok: Tokenizer) -> List[Int]:
    var tokens = List[Int]()
    for pos in range(len(text)):
        var char = str_to_ptr(text[pos])
        var tok_id = tok.find(char)
        if tok_id == -1:
            print("Not a good prompt token at pos ", pos)
            return tokens
        tokens.append(tok_id)
    while True:
        var best_score = Float32(-1e10)
        var best_id = -1
        var best_idx = -1

        for i in range(len(tokens) - 1):
            var str = str_concat(tok.vocab[tokens[i]], tok.vocab[tokens[i + 1]])
            var id = tok.find(str)
            var loaded_score = tok.vocab_scores.load(id)
            if id != -1 and loaded_score > best_score:
                best_score = loaded_score
                best_id = id
                best_idx = i

        if best_idx == -1:
            break

        tokens[best_idx] = best_id
        var _tokens = List[Int]()
        for i in range(0, best_idx + 1):
            _tokens.append(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            _tokens.append(tokens[i])
        tokens = _tokens
    return tokens

fn vector_to_matrix(vector: List[Int]) -> Matrix[float_dtype]:
    var total_size = len(vector)
    var out_matrix = Matrix[float_dtype](1, 1, total_size)

    @parameter
    fn vector_to_matrix_fn[width: Int](index: Int):
        var val = vector[index]
        var val_simd = SIMD[float_dtype, width].splat(int(val))
        out_matrix.store[width](0, 0, index, val_simd)

    vectorize[vector_to_matrix_fn, 1](total_size)
    return out_matrix

fn tensor_to_matrix(tensor: Tensor[float_dtype]) -> Matrix[float_dtype]:
    var out_matrix = Matrix[float_dtype](1,1, tensor.num_elements())

    @parameter
    fn tensor_to_matrix_fn[width: Int](index: Int):
        var val = tensor.load[width](index)
        out_matrix._data.store[width=width](index, val)

    vectorize[tensor_to_matrix_fn, simd_width](tensor.num_elements())
    return out_matrix

fn get_time_embedding(
    timestep:SIMD[float_dtype, 1]
) -> Matrix[float_dtype]:

    var freqs = Matrix[float_dtype](1, 1, 160)
    @parameter
    fn time_range_fn[width: Int](index: Int):
        var float_index: Float32 = index
        var val:Float32 = (-float_index / 160) ** 10000
        var val_simd = SIMD[float_dtype, width].splat(val)
        freqs.store[width](0, 0, index, val_simd)

    vectorize[time_range_fn, 1](160)

    var x = freqs * timestep
    var cos_x = x.cosine()
    var sin_x = x.sine()
    return cos_x.concat(sin_x, 2)

fn resize_image(
    image: Matrix[float_dtype], new_height: Int, new_width: Int
) -> Matrix[float_dtype]:
    var old_channels = image.dim0
    var old_width = image.dim1
    var old_height = image.dim2
    if old_height == new_height and old_width == new_width:
        return image

    var new_image = Matrix[float_dtype](old_channels, new_height, new_width)

    var scale_y = old_height / new_height
    var scale_x = old_width / new_width

    @parameter
    fn resize_channels(channel: Int):
        @parameter
        fn resize_row(row: Int):
            @parameter
            fn resize_image_fn[width: Int](col: Int):
                var new_y = int(row * scale_y)
                var new_x = int(col * scale_x)
                var val = image.load[1](channel, new_y, new_x)
                new_image.store[1](channel, row, col, val)

            vectorize[resize_image_fn, 1, unroll_factor=1](new_width)

        parallelize[resize_row](new_height, new_height)

    parallelize[resize_channels](old_channels, old_channels)
    return new_image

# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile_2d[tiled_fn: Tile2DFunc, stride_x: Int, stride_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, stride_y):
        for x in range(0, end_x, stride_x):
            tiled_fn[stride_x, stride_y](x, y)

fn Softmax(inout matrix: Matrix[float_dtype], dim:Int = 0) -> Matrix[float_dtype]:

    var exp_matrix = matrix.exp()

    if dim == 0:
        @parameter
        fn channel_softmax(channel: Int):
            var channel_sum = exp_matrix[channel, :, :].sum()
            var channel_div = exp_matrix[channel, :, :] / channel_sum
            exp_matrix.set_items(channel, Slice(0, matrix.dim1), Slice(0, matrix.dim2), channel_div)
        parallelize[channel_softmax](matrix.dim0, matrix.dim0)
        return exp_matrix
    elif dim == 1:
        @parameter
        fn row_softmax_channel(channel: Int):
            @parameter
            fn row_softmax[width: Int](row: Int):
                var row_sum = exp_matrix[channel, row, :].sum()
                var row_div = exp_matrix[channel, row, :] / row_sum
                exp_matrix.set_items(channel, row, Slice(0, matrix.dim2), row_div)
            vectorize[row_softmax, 1, unroll_factor=1](matrix.dim1)
        parallelize[row_softmax_channel](matrix.dim0, matrix.dim0)
        return exp_matrix

    elif dim == 2:
        @parameter
        fn column_softmax_channel(channel: Int):
            @parameter
            fn column_softmax[width: Int](column: Int):
                var col_sum = exp_matrix[channel, :, column].sum()
                var col_div = exp_matrix[channel, :, column] / col_sum
                exp_matrix.set_items(channel, Slice(0, matrix.dim1), column, col_div)
            vectorize[column_softmax, 1, unroll_factor=1](matrix.dim2)
        parallelize[column_softmax_channel](matrix.dim0, matrix.dim0)
        return exp_matrix
    else:
        print("Invalid dimension for softmax. Returning null matrix")
        return Matrix[float_dtype](0, 0, 0)

fn sigmoid(inout matrix: Matrix[float_dtype]) -> Matrix[float_dtype]:
    var new_matrix = matrix * -1
    new_matrix = new_matrix.exp()
    new_matrix += 1
    new_matrix = (new_matrix ** (-1))
    return new_matrix


struct Matrix_Array[dtype: DType]:
    var _data: DTypePointer[dtype]
    var matrix_shape: Tuple[Int, Int, Int]
    var matrix_size: Int
    var num_elements: Int

    fn __init__(inout self, num_elements: Int, matrix_shape: Tuple[Int, Int, Int]):
        self.matrix_shape = matrix_shape
        self.matrix_size = Tuple.get[0, Int](matrix_shape) * Tuple.get[1, Int](matrix_shape) * Tuple.get[2, Int](matrix_shape)
        self.num_elements = num_elements
        self._data = DTypePointer[dtype].alloc(self.matrix_size * num_elements)

    fn __copyinit__(inout self, other: Self):
        self._data = other._data
        self.matrix_shape = other.matrix_shape
        self.matrix_size = other.matrix_size
        self.num_elements = other.num_elements

    fn __setitem__(inout self, owned index: Int, new_el: Matrix[dtype]):
        var memory_index = index * self.matrix_size

        @parameter
        fn set_matrix(i : Int):
            self._data[memory_index + i] = new_el._data[i]

        parallelize[set_matrix](self.matrix_size, self.matrix_size)

    fn __setitem__(inout self, owned c_index: Int, owned z_index: Int, owned y_index: Int, owned x_index: Int, new_val: float_base):
        var memory_index = c_index * self.matrix_size + z_index * Tuple.get[1, Int](self.matrix_shape) * Tuple.get[2, Int](self.matrix_shape) + y_index * Tuple.get[2, Int](self.matrix_shape) + x_index
        var new_val_SIMD = SIMD[dtype, 1].splat(new_val.cast[dtype]())
        self._data[memory_index] = new_val_SIMD

    fn __getitem__(self, owned index: Int) -> Matrix[dtype]:
        var memory_index = index * self.matrix_size
        var dim0 = Tuple.get[0, Int](self.matrix_shape)
        var dim1 = Tuple.get[1, Int](self.matrix_shape)
        var dim2 = Tuple.get[2, Int](self.matrix_shape)
        var new_matrix = Matrix[dtype](dim0, dim1, dim2)

        @parameter
        fn get_matrix(i : Int):
            new_matrix._data[i] = self._data[memory_index + i]

        parallelize[get_matrix](self.matrix_size, self.matrix_size)

        return new_matrix

    fn __add__(self, other: Matrix[dtype]) -> Matrix_Array[dtype]:
        var out = Matrix_Array[dtype](self.num_elements, self.matrix_shape)
        @parameter
        fn add_fn(i: Int):
            out[i] = (self[i] + other)

        parallelize[add_fn](self.num_elements, self.num_elements)
        return out

    fn print(self):
        for i in range(self.num_elements):
            print("Matrix", i)
            self[i].print()

# Check out https://github.com/modularml/mojo/blob/main/examples/blogs-videos/mojo-matrix-Slice.ipynb
struct Matrix[dtype: DType]:
    var dim0: Int
    var dim1: Int
    var dim2: Int
    var _data: DTypePointer[dtype]

    fn __init__(inout self, *dims: Int):
        if dims[0] < 0 or dims[1] < 0 or dims[2] < 0:
            self.dim0 = 0
            self.dim1 = 0
            self.dim2 = 0
        self.dim0 = dims[0]
        self.dim1 = dims[1]
        self.dim2 = dims[2]
        self._data = DTypePointer[dtype].alloc(dims[0] * dims[1] * dims[2])
        rand(self._data, dims[0] * dims[1] * dims[2])

    fn init_weights(inout self, lower_bound: float_base, upper_bound:float_base):
        var low_bound = lower_bound.cast[DType.float64]()
        var up_bound = upper_bound.cast[DType.float64]()

        @parameter
        fn init_weights_fn[width: Int](index: Int) -> None:
            var weight_val = random_float64(low_bound, up_bound)
            var weight_simd = SIMD[DType.float64, width].splat(weight_val)
            var weight_simd_dtype = weight_simd.cast[dtype]()
            self._data.store[width=width](index, weight_simd_dtype)

        vectorize[init_weights_fn, 1](self.size().to_int())

    fn init_weights_normal(inout self, mean: float_base, std: float_base):
        var mean_val = mean.cast[DType.float64]()
        var std_val = std.cast[DType.float64]()

        @parameter
        fn init_weights_normal_fn[width: Int](index: Int) -> None:
            var weight_val = randn_float64(mean_val, std_val)
            var weight_simd = SIMD[DType.float64, width].splat(weight_val)
            var weight_simd_dtype = weight_simd.cast[dtype]()
            self._data.store[width=width](index, weight_simd_dtype)

        vectorize[init_weights_normal_fn, 1](self.size().to_int())

    fn init_weights_seed(inout self, seed_val: Int = 0):
        if seed_val == 0:
            seed()
        else:
            seed(seed_val)
        @parameter
        fn init_weights_random_fn[width: Int](index: Int) -> None:
            var weight_val = random_float64(1, 10000000)
            var weight_simd = SIMD[DType.float64, width].splat(weight_val)
            var weight_simd_dtype = weight_simd.cast[dtype]()
            self._data.store[width=width](index, weight_simd_dtype)

        vectorize[init_weights_random_fn, 1](self.size().to_int())

    fn rescale(inout self, old_scale: Tuple, new_scale: Tuple, clamp: Bool = False) -> Matrix[dtype]:
        var old_min = Tuple.get[0, Int](old_scale)
        var old_max = Tuple.get[1, Int](old_scale)
        var new_min = Tuple.get[0, Int](new_scale)
        var new_max = Tuple.get[1, Int](new_scale)

        var new_matrix = Matrix[dtype](self.dim0, self.dim1, self.dim2)

        @parameter
        fn rescale_fn[simd_width: Int](index: Int) -> None:
            var old_val = self._data.load[width=simd_width](index)
            var new_val_float = (old_val - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
            var new_val = new_val_float.cast[dtype]()
            new_matrix._data.store[width=simd_width](index, new_val)

        vectorize[rescale_fn, simd_width](self.size().to_int())

        if clamp:
            new_matrix = new_matrix.clamp(new_min, new_max)

        return new_matrix

    fn __copyinit__(inout self, other: Self):
        self._data = other._data
        self.dim0 = other.dim0
        self.dim1 = other.dim1
        self.dim2 = other.dim2

    fn concat(inout self, other: Self, dim: Int) -> Self:
        if dim < 0 or dim > 2:
            print("Invalid dimension for concatenation. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0 + other.dim0, self.dim1, self.dim2)

        if dim == 0:
            if self.dim1 != other.dim1 or self.dim2 != other.dim2:
                print("Non-matching dimensions for concatenation along the first axis. Returning null matrix")
                return Self(0, 0, 0)

            @parameter
            fn concat_fn0_self(c: Int):
                @parameter
                fn row_fn0_self(y: Int):
                    @parameter
                    fn col_fn0_self[simd_width: Int](x: Int):
                        var val = self.load[simd_width](c, y, x)
                        new_matrix.store[simd_width](c, y, x, val)

                    vectorize[col_fn0_self, simd_width, unroll_factor=simd_width](self.dim2)

                parallelize[row_fn0_self](self.dim1, self.dim1)

            parallelize[concat_fn0_self](self.dim0, self.dim0)

            @parameter
            fn concat_fn0_other(c: Int):
                @parameter
                fn row_fn0_other(y: Int):
                    @parameter
                    fn col_fn0_other[simd_width: Int](x: Int):
                        var val = other.load[simd_width](c, y, x)
                        new_matrix.store[simd_width](c + self.dim0, y, x, val)

                    vectorize[col_fn0_other, simd_width, unroll_factor=simd_width](other.dim2)

                parallelize[row_fn0_other](other.dim1, other.dim1)

            parallelize[concat_fn0_other](other.dim0, other.dim0)

            return new_matrix
        elif dim == 1:
            if self.dim0 != other.dim0 or self.dim2 != other.dim2:
                print("Non-matching dimensions for concatenation along the second axis. Returning null matrix")
                return Self(0, 0, 0)

            new_matrix = Self(self.dim0, self.dim1 + other.dim1, self.dim2)

            @parameter
            fn concat_fn1_self(c: Int):
                @parameter
                fn row_fn1_self(y: Int):
                    @parameter
                    fn col_fn1_self[simd_width: Int](x: Int):
                        var val = self.load[simd_width](c, y, x)
                        new_matrix.store[simd_width](c, y, x, val)

                    vectorize[col_fn1_self, simd_width, unroll_factor=simd_width](self.dim2)

                parallelize[row_fn1_self](self.dim1, self.dim1)

            parallelize[concat_fn1_self](self.dim0, self.dim0)

            @parameter
            fn concat_fn1_other(c: Int):
                @parameter
                fn row_fn1_other(y: Int):
                    @parameter
                    fn col_fn1_other[simd_width: Int](x: Int):
                        var val = other.load[simd_width](c, y, x)
                        new_matrix.store[simd_width](c, y + self.dim1, x, val)

                    vectorize[col_fn1_other, simd_width, unroll_factor=simd_width](other.dim2)

                parallelize[row_fn1_other](other.dim1, other.dim1)

            parallelize[concat_fn1_other](other.dim0, other.dim0)

        else:
            if self.dim0 != other.dim0 or self.dim1 != other.dim1:
                print("Non-matching dimensions for concatenation along the third axis. Returning null matrix")
                return Self(0, 0, 0)

            new_matrix = Self(self.dim0, self.dim1, self.dim2 + other.dim2)

            @parameter
            fn concat_fn2_self(c: Int):
                @parameter
                fn row_fn2_self(y: Int):
                    @parameter
                    fn col_fn2_self[simd_width: Int](x: Int):
                        var val = self.load[simd_width](c, y, x)
                        new_matrix.store[simd_width](c, y, x, val)

                    vectorize[col_fn2_self, simd_width, unroll_factor=simd_width](self.dim2)

                parallelize[row_fn2_self](self.dim1, self.dim1)

            parallelize[concat_fn2_self](self.dim0, self.dim0)

            @parameter
            fn concat_fn2_other(c: Int):
                @parameter
                fn row_fn2_other(y: Int):
                    @parameter
                    fn col_fn2_other[simd_width: Int](x: Int):
                        var val = other.load[simd_width](c, y, x)
                        new_matrix.store[simd_width](c, y, x + self.dim2, val)

                    vectorize[col_fn2_other, simd_width, unroll_factor=simd_width](other.dim2)

                parallelize[row_fn2_other](other.dim1, other.dim1)

            parallelize[concat_fn2_other](other.dim0, other.dim0)

        return new_matrix


    fn to_long(inout self) -> Matrix[DType.float64]:
        var new_matrix = Matrix[DType.float64](self.dim0, self.dim1, self.dim2)

        @parameter
        fn to_long_fn[width: Int](index: Int) -> None:
            var val = self._data.load[width=width](index)
            var val_long = val.cast[DType.float64]()
            new_matrix._data.store[width=width](index, val_long)

        vectorize[to_long_fn, simd_width](self.size().to_int())
        return new_matrix


    fn to_float32(inout self) -> Matrix[DType.float32]:
        var new_matrix = Matrix[DType.float32](self.dim0, self.dim1, self.dim2)

        @parameter
        fn to_float_fn[width: Int](index: Int) -> None:
            var val = self._data.load[width=width](index)
            var val_float32 = val.cast[DType.float32]()
            new_matrix._data.store[width=width](index, val_float32)

        vectorize[to_float_fn, simd_width](self.size().to_int())
        return new_matrix

    fn __adjust_Slice__(self, inout span: Slice, dim: Int) -> Slice:
        if span.start >= dim:
            span.start = dim - 1
        elif span.start < 0:
            span.start += dim
            if span.start < 0:
                span.start = 0
        if not span._has_end():
            span.end = dim
        elif span.end < 0:
            span.end += dim + 1
            if span.end < 0:
                span.end = 0
        elif span.end > dim:
            span.end = dim
        if span.end < span.start:
            span.start = 0
            span.end = 0
        return span

    fn __adjust_index(self, inout index: Int, dim: Int) -> Int:
        if index < 0:
            index += dim
            if index < 0:
                index = 0
        if index >= dim:
            index = dim - 1
        return index

    fn cosine(inout self) -> Matrix[float_dtype]:
        var new_matrix = Matrix[float_dtype](self.dim0, self.dim1, self.dim2)

        @parameter
        fn cosine_fn[width: Int](index: Int) -> None:
            var val = self._data.load[width=1](index)
            var val_simd = SIMD[DType.float32, 1].splat(val.cast[DType.float32]())
            var val_cosine = cos[float_dtype, 1](val_simd)
            new_matrix._data.store[width=1](index, val_cosine)

        vectorize[cosine_fn, 1](self.size().to_int())
        return new_matrix

    fn sine(inout self) -> Matrix[float_dtype]:
        var new_matrix = Matrix[float_dtype](self.dim0, self.dim1, self.dim2)

        @parameter
        fn sine_fn[width: Int](index: Int) -> None:
            var val = self._data.load[width=1](index)
            var val_simd = SIMD[DType.float32, 1].splat(val.cast[DType.float32]())
            var val_sine = sin[float_dtype, 1](val_simd)
            new_matrix._data.store[width=width](index, val_sine)

        vectorize[sine_fn, 1](self.size().to_int())
        return new_matrix

    fn load[simd_width: Int](self, z: Int, y: Int, x: Int) -> SIMD[dtype, simd_width]:
        var index = z * self.dim2 * self.dim1 + y * self.dim2 + x
        return self._data.load[width=simd_width](index)

    fn store[simd_width: Int](self, z:Int, y: Int, x: Int, val: SIMD[dtype, simd_width]):
        var index = z * self.dim2 * self.dim1 + y * self.dim2 + x
        return self._data.store[width=simd_width](index, val)

    fn __setitem__(self, owned z: Int, owned x: Int, owned y: Int, val: SIMD[dtype, 1]):
        z = self.__adjust_index(z, self.dim0)
        x = self.__adjust_index(x, self.dim1)
        y = self.__adjust_index(y, self.dim2)
        var val_simd = SIMD[dtype, 1].splat(val.cast[dtype]())
        self.store[1](z, x, y, val_simd)

    fn set_items(
        inout self, owned channel: Int, owned row: Int, col: Int, val: float_base
    ):
        self.set_items(
            Slice(channel, channel + 1), Slice(row, row + 1), Slice(col, col + 1), val
        )

    fn set_items(
        inout self,
        owned channel_Slice: Slice,
        owned row_Slice: Slice,
        col: Int,
        val: float_base,
    ):
        self.set_items(channel_Slice, row_Slice, Slice(col, col + 1), val)

    fn set_items(
        inout self,
        owned channel_Slice: Slice,
        row: Int,
        owned col_Slice: Slice,
        val: float_base,
    ):
        self.set_items(channel_Slice, Slice(row, row + 1), col_Slice, val)

    fn set_items(
        inout self, owned channel_Slice: Slice, row: Int, col: Int, val: float_base
    ):
        self.set_items(channel_Slice, Slice(row, row + 1), Slice(col, col + 1), val)

    fn set_items(
        inout self,
        channel: Int,
        owned row_Slice: Slice,
        owned col_Slice: Slice,
        val: float_base,
    ):
        self.set_items(Slice(channel, channel + 1), row_Slice, col_Slice, val)

    fn set_items(
        inout self, channel: Int, owned row_Slice: Slice, col: Int, val: float_base
    ):
        self.set_items(Slice(channel, channel + 1), row_Slice, Slice(col, col + 1), val)

    fn set_items(
        inout self, channel: Int, row: Int, owned col_Slice: Slice, val: float_base
    ):
        self.set_items(Slice(channel, channel + 1), Slice(row, row + 1), col_Slice, val)

    # Example usage: b.set_items(1,1,Slice(0,3), 7)
    fn set_items(
        inout self,
        owned channel_Slice: Slice,
        owned row_Slice: Slice,
        owned col_Slice: Slice,
        val: float_base,
    ):
        channel_Slice = self.__adjust_Slice__(channel_Slice, self.dim0)
        row_Slice = self.__adjust_Slice__(row_Slice, self.dim1)
        col_Slice = self.__adjust_Slice__(col_Slice, self.dim2)
        var val_simd = SIMD[dtype, 1].splat(val.cast[dtype]())

        @parameter
        fn Slice_channels_fn(channel_idx: Int):
            @parameter
            fn Slice_row_fn(row_idx: Int):
                @parameter
                fn Slice_col_fn[simd_width: Int](col_idx: Int) -> None:

                    self.store[simd_width](
                        channel_Slice[channel_idx],
                        row_Slice[row_idx],
                        col_Slice[0] + (col_idx),
                        val_simd,
                    )

                vectorize[Slice_col_fn, simd_width, unroll_factor=simd_width](
                    col_Slice.__len__()
                )

            parallelize[Slice_row_fn](row_Slice.__len__(), row_Slice.__len__())

        parallelize[Slice_channels_fn](channel_Slice.__len__(), channel_Slice.__len__())

    fn set_items(
        inout self,
        owned channel: Int,
        owned row: Int,
        col: Int,
        inout vals: Self,
    ) :
        self.set_items(
            Slice(channel, channel + 1), Slice(row, row + 1), Slice(col, col + 1), vals
        )

    fn set_items(
        inout self,
        owned channel_Slice: Slice,
        owned row_Slice: Slice,
        col: Int,
        inout vals: Self,
    ) :
        self.set_items(channel_Slice, row_Slice, Slice(col, col + 1), vals)

    fn set_items(
        inout self,
        owned channel_Slice: Slice,
        row: Int,
        owned col_Slice: Slice,
        inout vals: Self,
    ) :
        self.set_items(channel_Slice, Slice(row, row + 1), col_Slice, vals)

    fn set_items(
        inout self,
        owned channel_Slice: Slice,
        row: Int,
        col: Int,
        inout vals: Self,
    ) :
        self.set_items(channel_Slice, Slice(row, row + 1), Slice(col, col + 1), vals)

    fn set_items(
        inout self,
        channel: Int,
        owned row_Slice: Slice,
        owned col_Slice: Slice,
        inout vals: Self,
    ) :
        self.set_items(Slice(channel, channel + 1), row_Slice, col_Slice, vals)

    fn set_items(
        inout self,
        channel: Int,
        owned row_Slice: Slice,
        col: Int,
        inout vals: Self,
    ) :
        self.set_items(
            Slice(channel, channel + 1), row_Slice, Slice(col, col + 1), vals
        )

    fn set_items(
        inout self,
        channel: Int,
        row: Int,
        owned col_Slice: Slice,
        inout vals: Self,
    ) :
        self.set_items(
            Slice(channel, channel + 1), Slice(row, row + 1), col_Slice, vals
        )

    # Usage: b.set_items(Slice(0, 3), Slice(0, 3), Slice(0, 3), c)
    fn set_items(
        inout self,
        owned channel_Slice: Slice,
        owned row_Slice: Slice,
        owned col_Slice: Slice,
        inout vals: Self,
    ):
        channel_Slice = self.__adjust_Slice__(channel_Slice, self.dim0)
        row_Slice = self.__adjust_Slice__(row_Slice, self.dim1)
        col_Slice = self.__adjust_Slice__(col_Slice, self.dim2)

        if (
            channel_Slice.__len__() * row_Slice.__len__() * col_Slice.__len__()
            != vals.dim0 * vals.dim1 * vals.dim2
        ):
            return

        vals = vals.reshape(
            channel_Slice.__len__(), row_Slice.__len__(), col_Slice.__len__()
        )

        @parameter
        fn Slice_channels_fn(channel_idx: Int):
            @parameter
            fn Slice_rows_fn(row_idx: Int):
                @parameter
                fn Slice_cols_fn[simd_width: Int](idx: Int) -> None:
                    var vals_idx = vals._data.offset(
                        channel_idx * row_Slice.__len__() * col_Slice.__len__()
                        + row_idx * col_Slice.__len__()
                        + idx
                    )
                    var loaded_val = strided_load[dtype, simd_width](
                        vals_idx, col_Slice.step
                    )

                    self.store[simd_width](channel_Slice[channel_idx], row_Slice[row_idx], col_Slice[0] + (idx * col_Slice.step), loaded_val)

                vectorize[Slice_cols_fn, simd_width, unroll_factor=simd_width](
                    col_Slice.__len__()
                )

            parallelize[Slice_rows_fn](row_Slice.__len__(), row_Slice.__len__())

        parallelize[Slice_channels_fn](channel_Slice.__len__(), channel_Slice.__len__())

    # Usage: b.set_items(1,1,1), c)
    fn Slice_items(inout self, inout vals: Self) :
        self.set_items(
            Slice(0, self.dim0), Slice(0, self.dim1), Slice(0, self.dim2), vals
        )

    fn __getitem__(self, owned z: Int, owned x: Int, owned y: Int) -> SIMD[dtype, 1]:
        z = self.__adjust_index(z, self.dim0)
        x = self.__adjust_index(x, self.dim1)
        y = self.__adjust_index(y, self.dim2)
        var channel_adjustment = z * (self.dim1 * self.dim2)
        var row_adjustment = x * self.dim2
        return self._data.load[width=1](channel_adjustment + row_adjustment + y)

    fn __getitem__(
        self, owned channel_Slice: Slice, owned row_Slice: Slice, col: Int
    ) -> Self:
        return self.__getitem__(channel_Slice, row_Slice, Slice(col, col + 1))

    fn __getitem__(
        self, owned channel_Slice: Slice, row: Int, owned col_Slice: Slice
    ) -> Self:
        return self.__getitem__(channel_Slice, Slice(row, row + 1), col_Slice)

    fn __getitem__(self, owned channel_Slice: Slice, row: Int, col: Int) -> Self:
        return self.__getitem__(channel_Slice, Slice(row, row + 1), Slice(col, col + 1))

    fn __getitem__(
        self, channel: Int, owned row_Slice: Slice, owned col_Slice: Slice
    ) -> Self:
        return self.__getitem__(Slice(channel, channel + 1), row_Slice, col_Slice)

    fn __getitem__(self, channel: Int, owned row_Slice: Slice, col: Int) -> Self:
        return self.__getitem__(
            Slice(channel, channel + 1), row_Slice, Slice(col, col + 1)
        )

    fn __getitem__(self, channel: Int, row: Int, owned col_Slice: Slice) -> Self:
        return self.__getitem__(
            Slice(channel, channel + 1), Slice(row, row + 1), col_Slice
        )

    # Usage: a[:, 2:4, 7:]
    fn __getitem__(
        self, owned channel_Slice: Slice, owned row_Slice: Slice, owned col_Slice: Slice
    ) -> Self:
        channel_Slice = self.__adjust_Slice__(channel_Slice, self.dim0)
        row_Slice = self.__adjust_Slice__(row_Slice, self.dim1)
        col_Slice = self.__adjust_Slice__(col_Slice, self.dim2)

        var Sliced_mat = Self(
            channel_Slice.__len__(), row_Slice.__len__(), col_Slice.__len__()
        )

        @parameter
        fn Slice_channels_fn(channel_idx: Int):
            var channel_ptr = self._data.offset(
                channel_Slice[channel_idx] * self.dim1 * self.dim2
            )

            @parameter
            fn Slice_rows_fn(row_idx: Int):
                var row_ptr = channel_ptr.offset(
                    row_Slice[row_idx] * self.dim2 + col_Slice[0]
                )

                @parameter
                fn Slice_cols_fn[simd_width: Int](idx: Int):
                    var mat_idx = channel_idx * row_Slice.__len__() * col_Slice.__len__() + row_idx * col_Slice.__len__() + idx

                    var idx_pointer = row_ptr.offset(idx * col_Slice.step * simd_width)
                    var loaded_val = strided_load[dtype, simd_width](
                        idx_pointer, col_Slice.step
                    )
                    Sliced_mat._data.store[width=simd_width](mat_idx, loaded_val)

                vectorize[Slice_cols_fn, simd_width, unroll_factor=simd_width](
                    col_Slice.__len__()
                )

            parallelize[Slice_rows_fn](row_Slice.__len__(), row_Slice.__len__())

        parallelize[Slice_channels_fn](channel_Slice.__len__(), channel_Slice.__len__())

        return Sliced_mat

    fn size(self) -> float_base:
        return self.dim0 * self.dim1 * self.dim2

    fn print_dims(self) -> None:
        print(
            "Matrix:",
            self.dim0,
            "x",
            self.dim1,
            "x",
            self.dim2,
            ",",
            "DType:",
            dtype.__str__(),
        )

    fn reshape(inout self, dim0: Int, dim1: Int, dim2: Int)  -> Self:
        if dim0 * dim1 * dim2 != self.dim0 * self.dim1 * self.dim2:
            print("Invalid reshape dimensions that do not match the input size. Returning null matrix")
            return Self(0, 0, 0)

        if dim0 == self.dim0 and dim1 == self.dim1 and dim2 == self.dim2:
            return self

        if dim0 < 0 or dim1 < 0 or dim2 < 0:
            print("Invalid negative reshape dimensions. Returning null matrix")
            return Self(0, 0, 0)

        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2

        return self

    fn exp(self)  -> Self:
        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix.__copyinit__(self)
        var new_matrix_size = new_matrix.size().to_int()

        @parameter
        fn exp_fn[simd_width: Int](index: Int) -> None:
            new_matrix._data.store[width=simd_width](
                index, math.exp(self._data.load[width=simd_width](index))
            )

        vectorize[exp_fn, simd_width](new_matrix_size)

        return new_matrix

    fn sqrt(self)  -> Self:
        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix.__copyinit__(self)
        var new_matrix_size = new_matrix.size().to_int()

        @parameter
        fn sqrt_fn[simd_width: Int](index: Int) -> None:
            new_matrix._data.store[width=simd_width](
                index, math.sqrt(self._data.load[width=simd_width](index))
            )

        vectorize[sqrt_fn, simd_width](new_matrix_size)

        return new_matrix

    fn __mul__(self, y: float_base)  -> Self:
        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix.__copyinit__(self)
        var new_matrix_size = new_matrix.size().to_int()

        @parameter
        fn mul_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__mul__(y_simd)
            new_matrix._data.store[width=simd_width](index, computed_val)

        vectorize[mul_fn, simd_width](new_matrix_size)

        return new_matrix

    fn __imul__(self, y: float_base) -> None:
        @parameter
        fn mul_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__mul__(y_simd)
            self._data.store[width=simd_width](index, computed_val)

        vectorize[mul_fn, simd_width](self.size().to_int())

    fn __pow__(self, y: float_base)  -> Self:
        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix.__copyinit__(self)
        var new_matrix_size = new_matrix.size().to_int()

        @parameter
        fn pow_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__pow__(y_simd)
            new_matrix._data.store[width=simd_width](index, computed_val)

        vectorize[pow_fn, simd_width](new_matrix_size)

        return new_matrix

    fn __ipow__(self, y: float_base) -> None:
        @parameter
        fn pow_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__pow__(y_simd)
            self._data.store[width=simd_width](index, computed_val)

        vectorize[pow_fn, simd_width](self.size().to_int())

    fn __sub__(self, other: Self) -> Self:
        if self.dim0 != other.dim0 or self.dim1 != other.dim1 or self.dim2 != other.dim2:
            print("Non-matching dimensions for subtraction. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix *= 0

        @parameter
        fn channel_fn(c: Int):
            @parameter
            fn row_fn(y: Int):
                @parameter
                fn col_fn[simd_width: Int](x: Int):
                    var simd_val = self.load[simd_width](c, y, x)
                    var simd_val2 = other.load[simd_width](c, y, x)
                    var computed_val = simd_val.__sub__(simd_val2)
                    new_matrix.store[simd_width](c, y, x, computed_val)

                vectorize[col_fn, simd_width, unroll_factor=simd_width](self.dim2)

            parallelize[row_fn](self.dim1, self.dim1)

        parallelize[channel_fn](self.dim0, self.dim0)

        return new_matrix

    fn __add__(self, y: float_base)  -> Self:
        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix.__copyinit__(self)
        var new_matrix_size = new_matrix.size().to_int()

        @parameter
        fn add_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__add__(y_simd)
            new_matrix._data.store[width=simd_width](index, computed_val)

        vectorize[add_fn, simd_width](new_matrix_size)

        return new_matrix

    fn __add__(self, other: Self)  -> Self:
        if self.dim0 != other.dim0 or self.dim1 != other.dim1 or self.dim2 != other.dim2:
            print("Non-matching dimensions for addition. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix *= 0

        @parameter
        fn channel_fn(c: Int):
            @parameter
            fn row_fn(y: Int):
                @parameter
                fn col_fn[simd_width: Int](x: Int):
                    var simd_val = self.load[simd_width](c, y, x)
                    var simd_val2 = other.load[simd_width](c, y, x)
                    var computed_val = simd_val.__add__(simd_val2)
                    new_matrix.store[simd_width](c, y, x, computed_val)

                vectorize[col_fn, simd_width, unroll_factor=simd_width](self.dim2)

            parallelize[row_fn](self.dim1, self.dim1)

        parallelize[channel_fn](self.dim0, self.dim0)

        return new_matrix

    fn __iadd__(self, y: float_base) -> None:
        @parameter
        fn add_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__add__(y_simd)
            self._data.store[width=simd_width](index, computed_val)

        vectorize[add_fn, simd_width](self.size().to_int())

    fn __isub__(self, y: float_base) -> None:
        @parameter
        fn sub_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__add__(-y_simd)
            self._data.store[width=simd_width](index, computed_val)

        vectorize[sub_fn, simd_width](self.size().to_int())

    fn __truediv__(self, other: Self) -> Self:
        if self.dim0 != other.dim0 or self.dim1 != other.dim1 or self.dim2 != other.dim2:
            print("Non-matching dimensions for division. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix *= 0

        @parameter
        fn channel_fn(c: Int):
            @parameter
            fn row_fn(y: Int):
                @parameter
                fn col_fn[simd_width: Int](x: Int):
                    var simd_val = self.load[simd_width](c, y, x)
                    var simd_val2 = other.load[simd_width](c, y, x)
                    var computed_val = simd_val.__truediv__(simd_val2)
                    new_matrix.store[simd_width](c, y, x, computed_val)

                vectorize[col_fn, simd_width, unroll_factor=simd_width](self.dim2)

            parallelize[row_fn](self.dim1, self.dim1)

        parallelize[channel_fn](self.dim0, self.dim0)

        return new_matrix

    fn __truediv__(self, y: float_base)  -> Self:
        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix.__copyinit__(self)
        var new_matrix_size = new_matrix.size().to_int()

        @parameter
        fn div_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__truediv__(
                y_simd
            )
            new_matrix._data.store[width=simd_width](index, computed_val)

        vectorize[div_fn, simd_width](new_matrix_size)

        return new_matrix

    fn __itruediv__(self, y: float_base) -> None:
        @parameter
        fn div_fn[simd_width: Int](index: Int) -> None:
            var y_simd = SIMD[dtype, simd_width].splat(y.cast[dtype]())
            var computed_val = self._data.load[width=simd_width](index).__truediv__(
                y_simd
            )
            self._data.store[width=simd_width](index, computed_val)

        vectorize[div_fn, simd_width](self.size().to_int())

    fn sum(self) -> SIMD[dtype, 1]:
        var sum_simd = SIMD[dtype, 1].splat(0.0)

        for index in range(self.size().to_int()):
            sum_simd += self._data.load[width=1](index)

        return sum_simd

    fn mean(self) -> SIMD[dtype, 1]:
        return self.sum().__truediv__(self.size().cast[dtype]())

    # we use an unbiased estimator of the standard deviation
    fn std(self) -> SIMD[dtype, 1]:
        var mean = self.mean()
        var sum = self.sum()
        var sq_sum = SIMD[dtype, 1].splat(0.0)

        for i in range(self.size().to_int()):
            sq_sum += (self._data.load[width=1](i) - mean) ** 2

        return math.sqrt(sq_sum / self.size().cast[dtype]())

    # Order of padding is (top, bottom), (left, right)
    fn pad(self, padding_height: Tuple, padding_width: Tuple)  -> Self:
        var matrix_height = self.dim1
        var matrix_width = self.dim2
        var padding_height_top = Tuple.get[0, Int](padding_height)
        var padding_height_bottom = Tuple.get[1, Int](padding_height)
        var padding_width_left = Tuple.get[0, Int](padding_width)
        var padding_width_right = Tuple.get[1, Int](padding_width)
        var padded_width = (matrix_width + padding_width_left + padding_width_right)
        var padded_height = (matrix_height + padding_height_top + padding_height_bottom)
        var padded = Self(self.dim0, padded_height, padded_width)
        padded *= 0

        @parameter
        fn channel_fn(c: Int):
            @parameter
            fn row_fn(y: Int):
                @parameter
                fn col_fn[simd_width: Int](x: Int):
                    var c_simd = SIMD[dtype, simd_width].splat(c)
                    var x_simd = SIMD[dtype, simd_width].splat(x + padding_width_left)
                    var y_simd = SIMD[dtype, simd_width].splat(y + padding_height_top)
                    padded[c, y + padding_height_top, x + padding_width_left] = self[
                        c, y, x
                    ]

                vectorize[col_fn, 1, unroll_factor=1](matrix_width)

            parallelize[row_fn](matrix_height, matrix_height)

        parallelize[channel_fn](self.dim0, self.dim0)
        return padded

    # Elementwise multiplication. Usage:
    # var c = a.multiply(b)
    fn multiply(self, matrix: Self) -> Self:
        if self.dim0 != matrix.dim0 or self.dim1 != matrix.dim1 or self.dim2 != matrix.dim2:
            print("Non-matching dimensions for elementwise multiplication. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0, self.dim1, matrix.dim2)
        new_matrix *= 0

        @parameter
        fn channel_fn(c: Int):
            @parameter
            fn row_fn(y: Int):
                @parameter
                fn col_fn[simd_width: Int](x: Int):
                    var simd_val = self.load[simd_width](c, y, x)
                    var simd_val2 = matrix.load[simd_width](c, y, x)
                    var computed_val = simd_val.__mul__(simd_val2)
                    new_matrix.store[simd_width](c, y, x, computed_val)

                vectorize[col_fn, simd_width, unroll_factor=simd_width](self.dim2)

            parallelize[row_fn](self.dim1, self.dim1)

        parallelize[channel_fn](self.dim0, self.dim0)

        return new_matrix

    fn clamp(self, min_val: float_base, max_val: float_base) -> Self:
        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix *= 0

        @parameter
        fn clamp_fn[simd_width: Int](index: Int):
            var min_simd = SIMD[dtype, simd_width].splat(min_val.cast[dtype]())
            var max_simd = SIMD[dtype, simd_width].splat(max_val.cast[dtype]())
            var val = self._data.load[width=simd_width](index)
            var computed_val = val.max(min_simd).min(max_simd)
            new_matrix._data.store[width=simd_width](index, computed_val)

        vectorize[clamp_fn, simd_width, unroll_factor=simd_width](self.size().to_int())

        return new_matrix

    # Usage: var c = b.chunk(2, 2)
    fn chunk(self, chunk_dim: Int, num_chunks: Int) -> Matrix_Array[dtype]:
        if chunk_dim < 0 or chunk_dim  >= 3:
            print("Out of bounds chunk dimension. Returning null array")
            return Matrix_Array[dtype](0, (0,0,0))

        var chunk_axis = self.dim0
        if chunk_dim == 1:
            chunk_axis = self.dim1
        elif chunk_dim == 2:
            chunk_axis = self.dim2

        if num_chunks > chunk_axis:
            print("Number of chunks exceeds the size of the chunk axis. Returning null array")
            return Matrix_Array[dtype](0, (0,0,0))

        # For now, we only support chunking evenly for simplicity
        if chunk_axis % num_chunks != 0:
            print("Number of chunks does not evenly divide the size of the chunk axis. Returning null array")
            return Matrix_Array[dtype](0, (0,0,0))

        var chunk_size = chunk_axis // num_chunks
        var out_size = (chunk_size, self.dim1, self.dim2)
        if chunk_dim == 1:
            out_size = (self.dim0, chunk_size, self.dim2)
        elif chunk_dim == 2:
            out_size = (self.dim0, self.dim1, chunk_size)

        var out_array = Matrix_Array[dtype](num_chunks, out_size)

        @parameter
        fn chunk_fn(index: Int):
            var chunk_start = index * chunk_size
            var chunk_end = (index + 1) * chunk_size
            var chunk_Slice = Slice(chunk_start, chunk_end)
            if chunk_dim == 0:
                out_array[index] = self[chunk_Slice, Slice(0, self.dim1), Slice(0, self.dim2)]
            elif chunk_dim == 1:
                out_array[index] = self[Slice(0, self.dim0), chunk_Slice, Slice(0, self.dim2)]
            elif chunk_dim == 2:
                out_array[index] = self[Slice(0, self.dim0), Slice(0, self.dim1), chunk_Slice]

        parallelize[chunk_fn](num_chunks, num_chunks)

        return out_array


    # Usage: var d = b.transpose(0, 1) --> flips the coordinates for the 0 and 1 axes
    fn transpose(inout self, dim0: Int = 1, dim1: Int = 2) -> Self:
        if dim0 < 0 or dim0 >= 3 or dim1 < 0 or dim1 >= 3:
            print("Dimensions for transpose exceed matrix dimensions. Returning null matrix")
            return Self(0, 0, 0)

        if dim0 == dim1:
            return self

        # This covers the 0 and 1 case
        var new_matrix =  Self(self.dim1, self.dim0, self.dim2)
        if (dim0 == 0 and dim1 == 2) or (dim0 == 2 and dim1 == 0):
            new_matrix = Self(self.dim2, self.dim1, self.dim0)
        elif (dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1):
            new_matrix = Self(self.dim0, self.dim2, self.dim1)
        new_matrix *= 0

        @parameter
        fn transpose_fn[block_width: Int](index: Int):
            var x = index % self.dim2
            var y = (index // self.dim2) % self.dim1
            var z = index // (self.dim1 * self.dim2)
            var new_x = x
            var new_y = y
            var new_z = z
            if (dim0 == 0 and dim1 == 1) or (dim0 == 1 and dim1 == 0):
                new_z = y
                new_y = z
            if (dim0 == 0 and dim1 == 2) or (dim0 == 2 and dim1 == 0):
                new_x = z
                new_z = x
            if (dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1):
                new_y = x
                new_x = y

            new_matrix[new_z, new_y, new_x] = self[z, y, x]

        vectorize[transpose_fn, 1, unroll_factor=1](self.size().to_int())

        return new_matrix

    # This can be further optimized with tilingas you can check on the Mojo website. However, since I want to dynamically adjust the tile boundaries instead of assuming that the tile size will be a divisor of the tile function's boundaries, I didn't use it here.
    fn matmul(inout self, matrix: Self) -> Self:
        if self.dim2 != matrix.dim1:
            print("Non-matching dimensions for matrix multiplication. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0, self.dim1, matrix.dim2)
        new_matrix *= 0

        @parameter
        fn calc_channel(c: Int):
            @parameter
            fn calc_row(m: Int):
                for k in range(self.dim2):
                    @parameter
                    fn dot[simd_width : Int](n : Int):
                        new_matrix[c, m, n] += self[c, m, k] * matrix[c, k, n]
                    vectorize[dot, 1, unroll_factor=1](new_matrix.dim2)
            parallelize[calc_row](self.dim1, new_matrix.dim1)
        parallelize[calc_channel](self.dim0, new_matrix.dim0)

        return new_matrix

    fn triu(self, diagonal: Int = 0) -> Self:
        if diagonal != 0  and diagonal != 1:
            print("Invalid diagonal value. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix.__copyinit__(self)

        @parameter
        fn triu_channel(channel_idx: Int):
            @parameter
            fn triu_row(row_idx: Int):
                @parameter
                fn triu_col[width: Int](col_idx: Int):
                    if diagonal == 0:
                        if row_idx > col_idx:
                            new_matrix[channel_idx, row_idx, col_idx] = 0.0
                    else:
                        var adjusted_row_idx = new_matrix.dim1 - row_idx - 1

                        if row_idx < col_idx:
                            new_matrix[channel_idx, adjusted_row_idx, col_idx] = 0.0
                vectorize[triu_col, 1, unroll_factor=1](self.dim2)
            parallelize[triu_row](self.dim1, self.dim1)
        parallelize[triu_channel](self.dim0, self.dim0)

        return new_matrix

    fn masked_fill(self, mask: Self, value: float_base) -> Self:
        if self.dim0 != mask.dim0 or self.dim1 != mask.dim1 or self.dim2 != mask.dim2:
            print("Non-matching dimensions for masked fill. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0, self.dim1, self.dim2)
        new_matrix.__copyinit__(self)
        var simd_value = SIMD[dtype, 1].splat(value.cast[dtype]())

        @parameter
        fn masked_fill_channel(channel_idx: Int):
            @parameter
            fn masked_fill_row(row_idx: Int):
                @parameter
                fn masked_fill_col[width: Int](col_idx: Int):
                    if mask[channel_idx, row_idx, col_idx] != 0:
                        new_matrix[channel_idx, row_idx, col_idx] = simd_value
                vectorize[masked_fill_col, 1, unroll_factor=1](self.dim2)
            parallelize[masked_fill_row](self.dim1, self.dim1)
        parallelize[masked_fill_channel](self.dim0, self.dim0)

        return new_matrix

    fn broadcast_channel(self, dim1: Int, dim2: Int) -> Self:
        if self.dim1 != 1 and self.dim2 != 1:
            print("Non-scalar matrix cannot be broadcasted. Returning null matrix")
            return Self(0, 0, 0)

        var new_matrix = Self(self.dim0, dim1, dim2)
        new_matrix *= 0

        @parameter
        fn broadcast_channel(channel_idx: Int):
            @parameter
            fn broadcast_row(row_idx: Int):
                @parameter
                fn broadcast_col[width: Int](col_idx: Int):
                    new_matrix[channel_idx, row_idx, col_idx] = self[channel_idx, 0, 0]
                vectorize[broadcast_col, 1, unroll_factor=1](dim2)
            parallelize[broadcast_row](dim1, dim1)
        parallelize[broadcast_channel](self.dim0, self.dim0)

        return new_matrix

    fn print(self, prec: Int = 4) -> None:
        var dim0: Int = self.dim0
        var dim1: Int = self.dim1
        var dim2: Int = self.dim2
        var val: SIMD[dtype, 1] = 0.0

        if dim0 == 1 and dim1 == 1 and dim2 == 1:
            print(self[0, 0, 0])
            return

        if dim0 > 0 and dim1 > 0 and dim2 > 0:
            for i in range(dim0):
                for j in range(dim1):
                    if j == 0:
                        print("\n[\n", end="")
                    else:
                        print("\n", end="")
                    print(" [", end="")
                    for k in range(dim2):
                        val = self[i, j, k]
                        var int_str: String
                        if val > 0 or val == 0:
                            int_str = String(trunc(val).cast[DType.int32]())
                        else:
                            int_str = "-" + String(trunc(val).cast[DType.int32]())
                            val = -val
                        var float_str: String
                        float_str = String(mod(val, 1))
                        var s = int_str + "." + float_str[2 : prec + 2]
                        if k == 0:
                            print(s, end="")
                        else:
                            print("  ", s,end="")
                    print("]", end="")
                print("\n]\n", end="")
            print()
        print(
            "  Matrix:",
            self.dim0,
            "x",
            self.dim1,
            "x",
            self.dim2,
            ",",
            "DType:",
            dtype.__str__(),
        )
        print()


struct Conv2D:
    var out_channels: Int
    var in_channels: Int
    var kernel_size: Int
    var padding: Tuple[Int, Int]
    var stride: Tuple[Int, Int]
    var bias:Tensor[float_dtype]
    var kernel: Matrix_Array[float_dtype]

    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        padding: Tuple[Int, Int] = (0, 0),
        stride: Tuple[Int, Int] = (1, 1),
    )  -> None:
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        #### LEARNABLE PARAMETERS
        self.bias = Tensor[float_dtype](out_channels)
        self.kernel = Matrix_Array[float_dtype](out_channels, (in_channels, kernel_size, kernel_size))
        @parameter
        fn init_kernel_fn(out_channel_idx: Int):
            var curr_matrix = Matrix[float_dtype](self.in_channels, self.kernel_size, self.kernel_size)
            var k = (self.in_channels * self.kernel_size * self.kernel_size)
            var inv_k = math.rsqrt[float_dtype, 1](k)
            curr_matrix.init_weights(-inv_k, inv_k)
            self.kernel[out_channel_idx] = curr_matrix
        parallelize[init_kernel_fn](self.out_channels, self.out_channels)
        ####

    fn __copyinit__(inout self, other: Self) -> None:
        self.out_channels = other.out_channels
        self.in_channels = other.in_channels
        self.kernel_size = other.kernel_size
        self.padding = other.padding
        self.stride = other.stride
        self.bias = other.bias
        self.kernel = other.kernel

    fn forward(
        self,
        matrix: Matrix[float_dtype],
    )  -> Matrix[float_dtype]:

        var conv_matrix = matrix
        var padding_height = Tuple.get[0, Int](self.padding)
        var padding_width = Tuple.get[1, Int](self.padding)
        if Tuple.get[0, Int](self.padding) != 0 or Tuple.get[1, Int](self.padding) != 0:
            conv_matrix = conv_matrix.pad((padding_height, padding_height), (padding_width, padding_width))
        var height = conv_matrix.dim1
        var width = conv_matrix.dim2
        var stride_y = Tuple.get[0, Int](self.stride)
        var stride_x = Tuple.get[1, Int](self.stride)
        var final_height = math.floor(
            (height - self.kernel_size) / stride_y + 1
        ).to_int()

        var final_width = math.floor(
            (width - self.kernel_size) / stride_x + 1
        ).to_int()

        var output =
            Matrix[float_dtype](self.out_channels, final_height, final_width)

        @parameter
        fn channel_fn(out_channel_idx: Int):
            var kernel_channel = self.kernel[out_channel_idx]
            @parameter
            fn convolution_fn[stride_x: Int, stride_y: Int](x: Int, y: Int):
                var x_out = x // stride_x
                var y_out = y // stride_y
                var convolution_sum = SIMD[float_dtype, 1].splat(0.0)
                for in_channel_idx in range(self.in_channels):
                    var convolution_region = conv_matrix[
                        in_channel_idx,
                        y : y + self.kernel_size,
                        x : x + self.kernel_size,
                    ]
                    var kernel_region = kernel_channel[in_channel_idx,:,:]
                    var elementwise_mult = convolution_region.multiply(kernel_channel[in_channel_idx,:,:]).sum()

                    convolution_sum += elementwise_mult

                output[out_channel_idx, y_out, x_out] = convolution_sum + self.bias[out_channel_idx]

            var end_x = width - self.kernel_size + 1
            var end_y = height - self.kernel_size + 1

            # Here, we use these annoying if statements because the tiling function does not support dynamic values. Nonetheless, tiling gives a huge performance boost.
            if stride_x == 1 and stride_y == 1:
                tile_2d[convolution_fn, 1, 1](end_x, end_y)
            elif stride_x == 1 and stride_y == 0:
                tile_2d[convolution_fn, 1, 0](end_x, end_y)
            elif stride_x == 0 and stride_y == 1:
                tile_2d[convolution_fn, 0, 1](end_x, end_y)
            elif stride_x == 0 and stride_y == 0:
                tile_2d[convolution_fn, 0, 0](end_x, end_y)
            elif stride_x == 1 and stride_y == 2:
                tile_2d[convolution_fn, 1, 2](end_x, end_y)
            elif stride_x == 2 and stride_y == 1:
                tile_2d[convolution_fn, 2, 1](end_x, end_y)
            elif stride_x == 2 and stride_y == 2:
                tile_2d[convolution_fn, 2, 2](end_x, end_y)
            elif stride_x == 2 and stride_y == 0:
                tile_2d[convolution_fn, 2, 0](end_x, end_y)
            elif stride_x == 0 and stride_y == 2:
                tile_2d[convolution_fn, 0, 2](end_x, end_y)
            elif stride_x == 0 and stride_y == 0:
                tile_2d[convolution_fn, 0, 0](end_x, end_y)

        parallelize[channel_fn](self.out_channels, self.out_channels)

        return output

struct GroupNorm:
    var num_groups: Int
    var num_channels: Int
    var channels_per_group: Int
    var epsilon: float_base
    var gamma: float_base
    var beta: float_base

    fn __init__(
        inout self,
        num_groups: Int,
        num_channels: Int,
        epsilon: float_base = 1e-5,
    ) -> None:
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.channels_per_group = math.floor(num_channels / num_groups).to_int()
        self.epsilon = epsilon

        ### LEARNABLE PARAMETERS
        self.gamma = 1.0
        self.beta = 0.0
        ###

    fn __copyinit__(inout self, other: Self) -> None:
        self.num_groups = other.num_groups
        self.num_channels = other.num_channels
        self.channels_per_group = other.channels_per_group
        self.epsilon = other.epsilon
        self.gamma = other.gamma
        self.beta = other.beta

    fn forward(self, x: Matrix[float_dtype])  -> Matrix[float_dtype]:
        var output = Matrix[float_dtype](x.dim0, x.dim1, x.dim2)
        if self.num_channels > x.dim0:
            print("Number of channels exceeds the number of channels in the input matrix. Returning null matrix")
            return Matrix[float_dtype](0, 0, 0)

        if self.num_channels % self.num_groups != 0:
            print("Number of channels does not evenly divide the number of groups. Returning null matrix")
            return Matrix[float_dtype](0, 0, 0)

        @parameter
        fn channel_fn(i: Int):
            var channels_group = x[
                i * self.channels_per_group : (i + 1) * self.channels_per_group, :,:
            ]

            var mean = channels_group.mean()
            var std = channels_group.std()

            @parameter
            fn channels_per_group_fn(m: Int):
                @parameter
                fn compute_element[simd_width: Int](index: Int):
                    var channels_index = m * x.dim1 * x.dim2 + index
                    var curr_el = channels_group._data.load[width=simd_width](channels_index)

                    var el_normalized = (curr_el - mean) / (
                        std + self.epsilon
                    ) * self.gamma

                    var out_index = i * self.channels_per_group * x.dim1 * x.dim2 + m * x.dim1 * x.dim2 + index

                    output._data.store[width=simd_width](out_index, el_normalized)

                vectorize[compute_element, simd_width, unroll_factor=simd_width](x.dim1 * x.dim2)

            parallelize[channels_per_group_fn](self.channels_per_group, self.channels_per_group)

        parallelize[channel_fn](self.num_groups, self.num_groups)

        return output


struct SiLU:
    fn __init__(inout self) -> None:
        pass

    fn forward(self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var matrix = x

        @parameter
        fn vec_sigmoid[simd_width: Int](idx: Int) -> None:
            var x_idx = x._data.load[width=simd_width](idx)
            matrix._data.store[width=simd_width](idx, x_idx / (1 + math.exp(-x_idx)))

        vectorize[vec_sigmoid, simd_width, unroll_factor=simd_width](matrix.size().to_int())

        return matrix

struct Gelu:
    fn __init__(inout self) -> None:
        pass

    fn forward(self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var matrix = x

        @parameter
        fn vec_gelu[simd_width: Int](idx: Int) -> None:
            var x_idx = x._data.load[width=simd_width](idx)
            var cdf = 0.5 * (1 + math.tanh((math.sqrt[float_dtype, 1](2 / pi) * (x_idx + 0.044715 * x_idx ** 3))))
            matrix._data.store[width=simd_width](idx, x_idx * cdf)

        vectorize[vec_gelu, simd_width, unroll_factor=simd_width](matrix.size().to_int())

        return matrix

struct Linear:
    var in_features: Int
    var out_features: Int
    var bias: Matrix[float_dtype]
    var weight: Matrix[float_dtype]
    var use_bias: Bool

    fn __init__(
        inout self,
        in_features: Int,
        out_features: Int,
        use_bias : Bool = True,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        ### LEARNABLE PARAMETERS: bias and weight
        self.bias = Matrix[float_dtype](1, 1, out_features)
        var k = math.sqrt(self.in_features)
        var inv_k = math.rsqrt[float_dtype, 1](k)
        self.bias.init_weights(-inv_k, inv_k)
        self.weight = Matrix[float_dtype](1, out_features, in_features)
        self.weight.init_weights(-inv_k, inv_k)
        ###

    fn __copyinit__(inout self, other: Self) -> None:
        self.in_features = other.in_features
        self.out_features = other.out_features
        self.bias = other.bias
        self.weight = other.weight
        self.use_bias = other.use_bias

    fn forward(inout self, inout x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        if x.dim2 != self.in_features:
            print("Invalid input dimensions for Linear layer. Returning null matrix")
            return Matrix[float_dtype](0, 0, 0)

        var output = x.matmul(self.weight.transpose(1,2))

        if self.use_bias:
            var bias_matrix = Matrix[float_dtype](output.dim0, output.dim1, output.dim2)

            # Setting bias vectors in the same column to the same value
            @parameter
            fn channel_fn(i: Int):
                @parameter
                fn col_fn[width: Int](j: Int):
                    bias_matrix.set_items(i, Slice(0, bias_matrix.dim1), j, self.bias[0, 0, j])

                vectorize[col_fn, 1, unroll_factor=1]( bias_matrix.dim1)

            parallelize[channel_fn](output.dim0, output.dim0)
            output = output + bias_matrix

        return output


struct Upsample:
    var scale_factor: Int
    fn __init__(inout self, scale_factor:Int=1) -> None:
        if scale_factor < 1:
            print("Invalid scale factor for upsampling!")
        self.scale_factor = scale_factor

    fn __copyinit__(inout self, other: Self) -> None:
        self.scale_factor = other.scale_factor

    fn forward(self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:

        if self.scale_factor < 1:
            print("Invalid scale factor for upsampling. Returning null matrix")
            return Matrix[float_dtype](0, 0, 0)

        var new_channels = x.dim0 * self.scale_factor
        var output = Matrix[float_dtype](new_channels, x.dim1, x.dim2)

        @parameter
        fn channel_fn(i: Int):
            @parameter
            fn row_fn(j: Int):
                @parameter
                fn col_fn[simd_width: Int](k: Int):
                    var val = x.load[1](i // self.scale_factor, j, k)
                    output.store[1](i, j, k, val)
                vectorize[col_fn, 1, unroll_factor=1](x.dim2)

            parallelize[row_fn](x.dim1, x.dim1)

        return output



struct Embedding:
    var n_vocab: Int
    var n_embed: Int
    var weight: Matrix[float_dtype]

    fn __init__(inout self, n_vocab: Int, n_embed: Int) -> None:
        self.n_vocab = n_vocab
        self.n_embed = n_embed

        ## LEARNABLE PARAMETER: Weight
        self.weight = Matrix[float_dtype](1, n_vocab, n_embed)
        self.weight.init_weights_normal(0, 1)

    fn __copyinit__(inout self, other: Self) -> None:
        self.n_vocab = other.n_vocab
        self.n_embed = other.n_embed
        self.weight = other.weight

    fn forward(self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        var out = Matrix[float_dtype](1, x.dim2, self.n_embed)
        @parameter
        fn channel_fn(channel_idx: Int):
            @parameter
            fn row_fn(row_idx: Int):
                @parameter
                fn col_fn[width: Int](col_idx: Int):
                    var idx = int(x[channel_idx, row_idx, col_idx])
                    var weight_value = self.weight[0, idx, Slice(0, self.n_embed)]
                    out.set_items(channel_idx, row_idx, Slice(col_idx * self.n_embed, (col_idx + 1) * self.n_embed), weight_value)
                vectorize[col_fn, 1, unroll_factor=1](self.n_embed)
            parallelize[row_fn](x.dim2, x.dim2)
        parallelize[channel_fn](1, 1)
        return out

    fn print(self) -> None:
        self.weight.print()


struct LayerNorm:
    var group_norm: GroupNorm
    fn __init__(inout self, n_embed: Int) -> None:
        self.group_norm = GroupNorm(1, n_embed)

    fn __copyinit__(inout self, other: Self) -> None:
        self.group_norm = other.group_norm

    fn forward(self, x: Matrix[float_dtype]) -> Matrix[float_dtype]:
        return self.group_norm.forward(x)
