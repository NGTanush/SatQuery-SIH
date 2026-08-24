# RSVQA LoRA Adapter Run

- Date: 2026-08-24
- Base model: Salesforce/blip-vqa-base
- Dataset: local RSVQA subset at `datasets/rsvqa/train.jsonl`
- Output checkpoint: `checkpoints/rsvqa-blip-lora`
- Status: real fine-tuning run executed successfully on CUDA

## Training command used

```bash
python experiments/adaptation/train_lora.py \
  --train-jsonl datasets/rsvqa/train.jsonl \
  --output-dir checkpoints/rsvqa-blip-lora \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 5e-5 \
  --seed 42
```

## Hyperparameters

- seed: 42
- epochs: 1
- batch size: 1
- learning rate: 5e-5
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- target modules: query, value

## Result

The run executed on NVIDIA GeForce RTX 4050 Laptop GPU and completed successfully.

Observed output summary:

- Device: cuda
- PEFT model initialized
- Trainable parameters: 1,179,648
- Epoch 1/1 completed
- Average loss: 1.9846
- Training finished successfully!

## Artifact

Checkpoint directory:

- `checkpoints/rsvqa-blip-lora`

This contains the adapter weights and processor files required to load the PEFT model.

## Notes

- This is a real training run, not a placeholder.
- The project still requires separate held-out benchmark evaluation to report final accuracy and F1 metrics.
- The adapter is suitable for use in the project’s VQA pipeline when loaded through the PEFT model integration.
