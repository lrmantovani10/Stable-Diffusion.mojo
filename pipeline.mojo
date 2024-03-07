from helpers.utils import *
from clip import CLIP
from vae import Encoder, Decoder
from diffusion import Diffusion
from sampler import DDPMSampler


# We set the number of inference steps to 1, as we only want to do a single forward pass. Typical values would be around 50

# Also, this runs on a batch size of 1 (like in stochastic gradient descent. To use the same code but with a higher batch size, create a Matrix_Array struct (available in utils.mojo) and parallelize the generate() code for all its elements.
fn generate(
    prompt: String,
    backup_prompt: String = "",
    strength: Float32 = 0.8,
    cfg: Bool = True,
    cfg_scale: Float32 = 7.5,
    inference_steps: Int = 1,
    seed_val: Int = 0,
    input_image: Matrix[float_dtype] = Matrix[float_dtype](0, 0, 0),
) -> Matrix[float_dtype]:
    if (
        not SIMD[DType.float32, 1].splat(0.0)
        <= strength
        <= SIMD[DType.float32, 1].splat(1.0)
    ):
        print("Strength must be between 0 and 1. Returning empty matrix")
        return Matrix[float_dtype](0, 0, 0)

    var clip = CLIP()
    var tokenizer_ref = StringRef("tokenizer_clip.bin")
    var tokenizer_buffer = FileBuf()
    read_file(tokenizer_ref, tokenizer_buffer)

    # Using a vocab size of 49408, since we rely on the CLIP Tokenizer
    var tokenizer = Tokenizer(49408, tokenizer_buffer)
    var context: Matrix[float_dtype]
    var processed_prompt = prompt.replace(" ", "</w>")
    var processed_backup = backup_prompt.replace(" ", "</w>")
    if cfg:
        var prompt_tokens = DynamicVector[Int]()
        var cond_tokens_vector = bpe_encode(processed_prompt, tokenizer)
        var cond_tokens = vector_to_matrix(cond_tokens_vector)
        var cond_context = clip.forward(cond_tokens)
        var backup_tokens_vector = bpe_encode(processed_backup, tokenizer)
        var backup_tokens = vector_to_matrix(backup_tokens_vector)
        var backup_context = clip.forward(backup_tokens)
        context = cond_context.concat(backup_context, dim=0)
    else:
        var tokens_vector = bpe_encode(processed_prompt, tokenizer)
        var tokens = vector_to_matrix(tokens_vector)
        context = clip.forward(tokens)

    print("CLIP forward pass concluded")

    var sampler = DDPMSampler(seed_val)
    sampler.set_inference_timesteps(inference_steps)

    var latents_shape = (4, 64, 64)
    var latents = Matrix[float_dtype](
        Tuple.get[0, Int](latents_shape),
        Tuple.get[1, Int](latents_shape),
        Tuple.get[2, Int](latents_shape),
    )
    if input_image.size() > 0:
        var encoder = Encoder()
        print("Encoder instance created")
        var resized_input = resize_image(input_image, 512, 512)
        var rescaled_input = resized_input.rescale((0, 255), (-1, 1))
        var encoder_noise = Matrix[float_dtype](
            Tuple.get[0, Int](latents_shape),
            Tuple.get[1, Int](latents_shape),
            Tuple.get[2, Int](latents_shape),
        )
        encoder_noise.init_weights_seed(seed_val)
        print("Encoder noise initialized")
        latents = encoder.forward(rescaled_input, encoder_noise)
        sampler.set_strength(strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])
    else:
        latents.init_weights_seed(seed_val)

    var diffusion = Diffusion()
    print("Diffusion instance created")

    var num_timesteps = sampler.timesteps.num_elements()
    for i in range(num_timesteps):
        var timestep = sampler.timesteps[i]
        var time_embedding = get_time_embedding(timestep)
        var model_input = latents
        if cfg:
            model_input = model_input.concat(model_input, dim=0)

        var model_output = diffusion.forward(model_input, context, time_embedding)

        if cfg:
            var chunked_output = model_output.chunk(0, 2)
            var conditional_output = chunked_output[0]
            var backup_output = chunked_output[1]
            var cfg_scale_f32 = SIMD[DType.float32, 1].splat(
                cfg_scale.cast[DType.float32]()
            )
            model_output = (
                conditional_output - backup_output
            ) * cfg_scale_f32 + backup_output

        latents = sampler.step(int(timestep), latents, model_output)
        print("Timestep", i, "concluded")

    var decoder = Decoder()
    var images = decoder.forward(latents)
    print("Decoder forward pass concluded")
    images = images.rescale((-1, 1), (0, 255), clamp=True)
    return images
