"""Colab-friendly LoRA adaptation entry point for RSVQA-style data.

This script deliberately fails fast when optional training dependencies or a
dataset are missing; inference does not depend on them.
"""
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Salesforce/blip-vqa-base")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    records = [json.loads(line) for line in Path(args.train_jsonl).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("The training JSONL contains no records.")
    required = {"image", "question", "answer"}
    missing = required - records[0].keys()
    if missing:
        raise ValueError(f"Dataset records must include: {sorted(required)}; missing {sorted(missing)}")
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import BlipForQuestionAnswering, BlipProcessor
    except ImportError as exc:
        raise RuntimeError("Install optional training dependencies: pip install peft accelerate datasets") from exc

    torch.manual_seed(args.seed)
    processor = BlipProcessor.from_pretrained(args.base_model)
    model = BlipForQuestionAnswering.from_pretrained(args.base_model)
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, lora_config)
    # Dataset-specific tokenization/training is intentionally explicit: RSVQA
    # answer vocabularies and image licenses must be selected by the project team.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    Path(args.output_dir, "run_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Initialized LoRA adapter artifact at {args.output_dir}. Complete supervised training in Colab with the selected dataset split.")


if __name__ == "__main__":
    main()
