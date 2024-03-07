import pipeline
from helpers.utils import *
fn main() raises -> None:
    let prompt = "a cat flying a spaceship"
    let backup_prompt = ''
    let input_image = Matrix[float_dtype](1, 32, 32)
    let do_cfg = True
    let cfg_scale = 0.8
    let strength = 0.9
    let num_inference_steps = 1
    let seed = 40

    let output_image = pipeline.generate(
    prompt=prompt,
    backup_prompt=backup_prompt,
    strength=strength,
    cfg=do_cfg,
    cfg_scale=cfg_scale,
    inference_steps=num_inference_steps,
    seed_val=seed,
    input_image = input_image
)