import pipeline

fn main() raises -> None:
    let prompt = "a cat flying a spaceship"
    let backup_prompt = ''
    let input_image = None
    let do_cfg = False
    let cfg_scale = 0.8
    let strength = 0.9
    let num_inference_steps = 50
    let seed = 40

    let output_image = pipeline.generate(
    prompt=prompt,
    backup_prompt=backup_prompt,
    strength=strength,
    cfg=do_cfg,
    cfg_scale=cfg_scale,
    inference_steps=num_inference_steps,
    seed_val=seed,
)