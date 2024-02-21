from helpers.utils import *
from clip import CLIP
from vae import Encoder, Decoder
from diffusion import Diffusion
from sampler import DDPMSampler


fn generate(
    prompt: String,
    backup_prompt: String = "",
    strength: Float32 = 0.8,
    cfg: Bool = True,
    cfg_scale: Float32 = 7.5,
    inference_steps: Int = 50,
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
    let processed_prompt = prompt.replace(" ", "</w>")
    let processed_backup = backup_prompt.replace(" ", "</w>")
    if cfg:
        var prompt_tokens = DynamicVector[Int]()
        let cond_tokens_vector = bpe_encode(processed_prompt, tokenizer)
        var cond_tokens = vector_to_matrix(cond_tokens_vector)
        var cond_context = clip.forward(cond_tokens)
        let backup_tokens_vector = bpe_encode(processed_backup, tokenizer)
        var backup_tokens = vector_to_matrix(backup_tokens_vector)
        let backup_context = clip.forward(backup_tokens)
        context = cond_context.concat(backup_context, dim=0)
    else:
        let tokens_vector = bpe_encode(processed_prompt, tokenizer)
        var tokens = vector_to_matrix(tokens_vector)
        context = clip.forward(tokens)

    print("CLIP forward pass concluded")

    var sampler = DDPMSampler(seed_val)
    sampler.set_inference_timesteps(inference_steps)

    let latents_shape = Tensor[DType.int64](4, 64, 64)
    var latents = Matrix[float_dtype](
        int(latents_shape[0]), int(latents_shape[1]), int(latents_shape[2])
    )
    if input_image.size() > 0:
        var encoder = Encoder()
        var resized_input = resize_image(input_image, 512, 512)
        let rescaled_input = resized_input.rescale((0, 255), (-1, 1))
        var encoder_noise = Matrix[float_dtype](
            int(latents_shape[0]), int(latents_shape[1]), int(latents_shape[2])
        )
        encoder_noise.init_weights_seed(seed_val)
        latents = encoder.forward(rescaled_input, encoder_noise)
        sampler.set_strength(strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])
    else:
        latents.init_weights_seed(seed_val)
    var diffusion = Diffusion()
    let num_timesteps = sampler.timesteps.num_elements()
    for i in range(num_timesteps):
        let timestep = sampler.timesteps[i]
        var time_embedding = get_time_embedding(timestep)
        print("CHECKPOINT - time embedding generated")
        var model_input = latents
        if cfg:
            model_input = model_input.concat(model_input, dim=0)
        var model_output = diffusion.forward(model_input, context, time_embedding)
        print("Diffusion forward pass concluded")

        if cfg:
            let chunked_output = model_output.chunk(0, 2)
            let conditional_output = chunked_output[0]
            let backup_output = chunked_output[1]
            let cfg_scale_f32 = SIMD[DType.float32, 1].splat(
                cfg_scale.cast[DType.float32]()
            )
            model_output = (
                conditional_output - backup_output
            ) * cfg_scale_f32 + backup_output

        latents = sampler.step(int(timestep), latents, model_output)

    var decoder = Decoder()
    var images = decoder.forward(latents)
    print("Decoder forward pass concluded")
    images = images.rescale((-1, 1), (0, 255), clamp=True)
    return images[0, :, :]
