import pipeline
from helpers.utils import *
fn main() raises -> None:
    var prompt = "a cat flying a spaceship"
    var backup_prompt = ''
    var input_image = Matrix[float_dtype](1, 8, 8)
    var do_cfg = False
    var cfg_scale = 0.8
    var strength = 0.9
    var num_inference_steps = 1
    var seed = 40

    var output_image = pipeline.generate(
    prompt=prompt,
    backup_prompt=backup_prompt,
    strength=strength,
    cfg=do_cfg,
    cfg_scale=cfg_scale,
    inference_steps=num_inference_steps,
    seed_val=seed,
    input_image = input_image
)
