from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Anima LoRA sample image with DiffSynth-Studio.")
    parser.add_argument("--dit", required=True)
    parser.add_argument("--qwen3", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--qwen_tokenizer", default="")
    parser.add_argument("--sd35_tokenizer", default="")
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low_vram", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig

    for path in (args.dit, args.qwen3, args.vae, args.lora):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    for path in (args.qwen_tokenizer, args.sd35_tokenizer):
        if path and not Path(path).exists():
            raise FileNotFoundError(path)

    vram_config = {}
    if args.low_vram:
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

    pipe = AnimaImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=args.dit, **vram_config),
            ModelConfig(path=args.qwen3, **vram_config),
            ModelConfig(path=args.vae, **vram_config),
        ],
        tokenizer_config=ModelConfig(args.qwen_tokenizer)
        if args.qwen_tokenizer
        else ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="./"),
        tokenizer_t5xxl_config=ModelConfig(args.sd35_tokenizer)
        if args.sd35_tokenizer
        else ModelConfig(model_id="stabilityai/stable-diffusion-3.5-large", origin_file_pattern="tokenizer_3/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5 if torch.cuda.is_available() else None,
    )
    pipe.load_lora(pipe.dit, args.lora)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=int(args.width),
        height=int(args.height),
        seed=int(args.seed),
        cfg_scale=float(args.cfg_scale),
        num_inference_steps=int(args.steps),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
