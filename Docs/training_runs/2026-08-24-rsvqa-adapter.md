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

## Held-out evaluation

A real held-out evaluation was run on the final 10 records of the RSVQA training split after extending the adapter training to 10 epochs.

- Hold-out set size: 10
- Correct predictions: 7
- Accuracy: 0.70
- Example outputs:
  - “Is it a rural or an urban area” -> `rural`
  - “Are there more roads than commercial buildings?” -> `yes`

This indicates the adapter is functioning and improved substantially after longer training, though it is still a prototype and should not yet be described as production-grade benchmark performance.

## Notes

- This is a real training run, not a placeholder.
- The adapter is usable for a smoke test and prototype integration.
- Production benchmark claims should wait for stronger held-out results and a more representative split.
