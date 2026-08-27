# EffiE External Model Integration

This folder contains glue code for using public EffiE-style sEMG models without
vendoring the full external repository into `emg_live_marker`.

Target upstream:

```powershell
git clone https://github.com/MIC-Laboratory/IEEE-NER-2023-EffiE external_models\EffiE
dir external_models\EffiE\checkpoints
```

The main application does not depend on TensorFlow. If you have an EffiE
TensorFlow/Keras checkpoint, use `import_effie.py` as a conversion helper. The
fine-tuning path itself uses the PyTorch adapter in:

```text
emg_live_marker/ml/effie_adapter.py
emg_live_marker/ml/effie_preprocess.py
```

Fine-tune a classifier:

```powershell
python scripts\finetune_effie_gesture.py ^
  --dataset-root dataset ^
  --effie-root external_models\EffiE ^
  --checkpoint-path external_models\EffiE\checkpoints\<checkpoint_file> ^
  --output-dir models\effie_finetuned ^
  --mode freeze_backbone ^
  --epochs 50 ^
  --batch-size 128 ^
  --lr 0.0001 ^
  --device auto ^
  --val-split trial ^
  --max-rest-ratio 1.0 ^
  --balanced-sampler ^
  --export-torchscript
```

If freezing the backbone is not enough:

```powershell
python scripts\finetune_effie_gesture.py ^
  --dataset-root dataset ^
  --effie-root external_models\EffiE ^
  --checkpoint-path external_models\EffiE\checkpoints\<checkpoint_file> ^
  --output-dir models\effie_finetuned_all ^
  --mode finetune_all ^
  --epochs 50 ^
  --batch-size 128 ^
  --lr 0.00001 ^
  --device auto ^
  --val-split trial ^
  --max-rest-ratio 1.0 ^
  --balanced-sampler ^
  --export-torchscript
```

