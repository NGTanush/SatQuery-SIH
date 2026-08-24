"""LoRA adaptation entry point for training on RSVQA-style data."""
import argparse
import json
import re
from pathlib import Path
from PIL import Image


def normalize_answer(raw):
    """Normalize answer text so equivalent labels are treated consistently."""
    if raw is None:
        return ""
    text = str(raw).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,!?:;\"'[](){}<>")
    if text:
        text = text.replace("’", "'")
        text = text.replace("-", " ")
        text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Salesforce/blip-vqa-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
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

    records = [{
        "image": rec["image"],
        "question": str(rec["question"]).strip(),
        "answer": normalize_answer(rec["answer"]),
    } for rec in records]
    
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from peft import LoraConfig, get_peft_model
        from transformers import BlipForQuestionAnswering, BlipProcessor
    except ImportError as exc:
        raise RuntimeError("Install optional training dependencies: pip install peft accelerate datasets") from exc

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model and processor...")
    processor = BlipProcessor.from_pretrained(args.base_model)
    model = BlipForQuestionAnswering.from_pretrained(args.base_model)
    
    # Configure LoRA. BlipForQuestionAnswering decoder attention layers have query/value projections.
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        bias="none", 
        target_modules=["query", "value"]
    )
    
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()
    
    print(f"PEFT model initialized. Trainable parameters:")
    model.print_trainable_parameters()

    # Custom Dataset class
    class RSVQADataSet(Dataset):
        def __init__(self, data_records):
            self.records = data_records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            img_path = rec["image"]
            question = rec["question"]
            answer = rec["answer"]
            
            image = Image.open(img_path).convert("RGB")
            return image, question, answer

    # Collate function to tokenize and batch
    def collate_fn(batch):
        images = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        
        inputs = processor(images=images, text=questions, return_tensors="pt", padding=True)
        
        labels = processor(text=answers, return_tensors="pt", padding=True).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        return inputs

    dataset = RSVQADataSet(records)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("Starting training loop...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move inputs to device
            batch_inputs = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch_inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (step + 1) % 5 == 0 or step == len(dataloader) - 1:
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} completed. Average Loss: {avg_loss:.4f}")

    print(f"Saving fine-tuned adapter to {args.output_dir}...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Save the run config
    Path(args.output_dir, "run_config.json").write_text(json.dumps(vars(args), indent=2))
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
