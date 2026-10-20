# Robot Checkpoint

This directory contains the final DROID-fine-tuned Slot-JEPA checkpoint used by robot planning and as the frozen visual backbone for representation evaluation.

| File | Purpose |
| --- | --- |
| `slot_jepa_droid_finetuned.pt` | Final DROID-fine-tuned model weights |
| `slot_jepa_droid_finetuned.yaml` | Minimal architecture and loading configuration |

From `workflows/robot/`, start the model server with:

```bash
python serve_slot.py \
  --ckpt-path ../../artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.pt \
  --config-path ../../artifacts/checkpoints/downstream/robot/slot_jepa_droid_finetuned.yaml
```

Run the following from this directory to verify the full downstream checkpoint package:

```bash
(cd .. && sha256sum -c SHA256SUMS)
```
