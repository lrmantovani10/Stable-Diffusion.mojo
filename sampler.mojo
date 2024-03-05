from helpers.utils import *
from math import round


struct DDPMSampler:
    var seed_val: Int
    var num_training_steps: Int
    var betas: Tensor[float_dtype]
    var alphas: Tensor[float_dtype]
    var alphas_cumprod: Tensor[float_dtype]
    var timesteps: Tensor[float_dtype]
    var num_inference_steps: Int
    var start_step: Int

    fn __init__(
        inout self,
        seed_val: Int = 0,
        # Setting this to 10 for illustrative purposes, since we are not interested in training. Typical values would be around 1000
        num_training_steps: Int = 10,
        beta_start: Float32 = 0.00085,
        beta_end: Float32 = 0.0120,
    ):
        # Setting this to 1 since I am intersted in demonstrating a single forward pass
        self.num_inference_steps = 1
        self.start_step = 0
        self.seed_val = seed_val
        self.num_training_steps = num_training_steps
        self.betas = (
            linspace(beta_start**0.5, beta_end**0.5, num_training_steps) ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = cumprod(self.alphas)
        self.timesteps = arange(0, num_training_steps, True)

    fn set_inference_timesteps(
        inout self,
        num_inference_steps: Int = 1,
    ):
        self.num_inference_steps = num_inference_steps
        let step_ratio: Float32 = self.num_training_steps // self.num_inference_steps
        let timesteps = round_tensor(
            arange(0, self.num_inference_steps, True) * step_ratio
        )
        self.timesteps = timesteps

    fn get_previous_timestep(
        inout self,
        timestep: Int,
    ) -> Int:
        let prev_t = timestep - self.num_training_steps // self.num_inference_steps
        return prev_t

    fn get_variance(
        inout self,
        timestep: Int,
    ) -> Float32:
        let prev_t = self.get_previous_timestep(timestep)
        let alpha_prod_t = self.alphas_cumprod[timestep]
        let alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1
        let current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        var variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # Preventing zero values
        variance = variance.max(1e-20)
        return variance[0]

    fn set_strength(inout self, strength: Float32):
        let start_step = self.num_inference_steps - int(
            self.num_inference_steps * strength
        )
        let timesteps_length = self.timesteps.num_elements()
        self.timesteps = get_tensor_values(self.timesteps, start_step, timesteps_length)
        self.start_step = start_step

    fn step(
        inout self,
        timestep: Int,
        latents: Matrix[float_dtype],
        model_output: Matrix[float_dtype],
    ) -> Matrix[float_dtype]:
        let prev_t = self.get_previous_timestep(timestep)
        let alpha_prod = self.alphas_cumprod[timestep]
        let alpha_prod_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1
        let beta_prod = 1 - alpha_prod
        let beta_prod_prev = 1 - alpha_prod_prev
        let current_alpha = alpha_prod / alpha_prod_prev
        let current_beta = 1 - current_alpha
        let alpha_prod_final = alpha_prod[0]
        let beta_prod_final = beta_prod[0]
        let pred_original_sample = (
            latents - (model_output) * (beta_prod_final ** (0.5))
        ) / (alpha_prod_final ** (0.5))
        let pred_original_sample_coefficient: Float32 = (
            alpha_prod_prev ** (0.5) * current_beta
        ) / beta_prod
        let current_sample_coefficient = current_alpha ** (
            0.5
        ) * beta_prod_prev / beta_prod
        var pred_previous_sample = pred_original_sample * pred_original_sample_coefficient + latents * current_sample_coefficient

        if timestep > 0:
            var noise = Matrix[float_dtype](
                model_output.dim0, model_output.dim1, model_output.dim2
            )
            noise.init_weights_seed(self.seed_val)
            let multiplier = (self.get_variance(timestep) ** 0.5)
            let variance = noise * multiplier
            pred_previous_sample = pred_previous_sample + variance
        return pred_previous_sample

    fn add_noise(
        inout self, original_samples: Matrix[float_dtype], timestep: Float32
    ) -> Matrix[float_dtype]:
        let int_timestep = int(timestep)
        let sqrt_alpha_prod = self.alphas_cumprod[int_timestep] ** 0.5
        var sqrt_alpha_prod_matrix = Matrix[float_dtype](1, 1, 1)
        sqrt_alpha_prod_matrix[0, 0, 0] = sqrt_alpha_prod[0]
        let sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[int_timestep]) ** 0.5
        var sqrt_one_minus_alpha_prod_matrix = Matrix[float_dtype](1, 1, 1)
        sqrt_one_minus_alpha_prod_matrix[0, 0, 0] = sqrt_one_minus_alpha_prod[0]
        var noise = Matrix[float_dtype](
            original_samples.dim0, original_samples.dim1, original_samples.dim2
        )
        noise.init_weights_seed(self.seed_val)
        let noisy_samples = sqrt_alpha_prod_matrix.multiply(
            original_samples
        ) + sqrt_one_minus_alpha_prod_matrix.multiply(noise)
        return noisy_samples
