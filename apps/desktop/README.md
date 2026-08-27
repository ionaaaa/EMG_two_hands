# emg_live_marker

`emg_live_marker` is a Python desktop app for real-time display of 8-channel Bluetooth transparent-serial EMG wristband data, with event marking, simulation, serial connection, and session recording.

It supports simulated data for development without hardware and real serial input through pyserial.

## Layout

```text
src/emg_live_marker/  # installable package
scripts/              # thin launch wrappers
tests/                # automated tests
```

## Install

```powershell
$python = "/Applications/anaconda3/envs/BN5213/bin/python"
& $python -m pip install -e ".[dev]"
```

The project standard is the `BN5213` Conda environment. Do not rely on the
checked-in Windows-format `venv/`; create another environment only when that is
explicitly intended. The editable installation is required before using the
thin scripts or package CLI from outside this directory.

## Test

```powershell
& $python -m pytest
```

## Run

```powershell
& $python -m emg_live_marker --simulate
& $python -m emg_live_marker --port COM4
```

No hardware is required for the simulator:

```powershell
& $python -m emg_live_marker --simulate
```

Running without arguments also starts the simulator:

```powershell
& $python -m emg_live_marker
```

Real serial mode on Windows:

```powershell
& $python -m emg_live_marker --port COM4 --baudrate 921600
```

## Paths

All project paths are resolved from the repository root, not from the current
working directory. Defaults are `data/datasets`, `data/recordings`, and
`apps/desktop` for artifacts. Use `--dataset-root`, `--recordings-root`,
and `--artifacts-root` to override them. A local repository-root
`.emg-paths.json` can provide the same three keys and is ignored by Git; its
values override environment variables (`EMG_DATASET_ROOT`, `EMG_RECORDINGS_ROOT`,
`EMG_ARTIFACTS_ROOT`) but are overridden by command-line options.

## Calibration Game Model

For live game demos, cross-session accuracy can be low because bracelet placement
changes between sessions. After wearing the bracelet, collect a small calibration
set for the three gestures, then train a same-day game model:

```powershell
& $python scripts/train_gesture_classifier.py ^
  --dataset-root data/datasets ^
  --preset calibration ^
  --device auto
```

The calibration preset trains the `emg2pose_tcn` classifier with balanced rest
windows, balanced sampling/loss, best-checkpoint saving, and TorchScript export.
The Game Decoder will prefer `apps/desktop/models/calibration_game_model/gesture_classifier.ts`
when it exists.

## EffiE Transfer Learning

`emg2pose` checkpoints are not used directly because they are 2kHz, 16-channel
pose-regression models. For a public model closer to this hardware, use the
Myo/8-channel/200Hz EffiE project:

```powershell
git clone https://github.com/MIC-Laboratory/IEEE-NER-2023-EffiE ..\..\third_party\EffiE
dir ..\..\third_party\EffiE\checkpoints
```

Freeze the EffiE-style backbone and train a 4-class game head:

```powershell
& $python scripts\finetune_effie_gesture.py ^
  --dataset-root data/datasets ^
  --effie-root ..\..\third_party\EffiE ^
  --checkpoint-path ..\..\third_party\EffiE\checkpoints\<checkpoint_file> ^
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

If freezing the backbone is not enough, fine-tune the whole model with a lower
learning rate:

```powershell
& $python scripts\finetune_effie_gesture.py ^
  --dataset-root data/datasets ^
  --effie-root ..\..\third_party\EffiE ^
  --checkpoint-path ..\..\third_party\EffiE\checkpoints\<checkpoint_file> ^
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

The exported `gesture_classifier.ts` uses EffiE-style preprocessing: raw 250Hz
EMG is resampled to 200Hz, the latest 32 samples are used as an `8x32` window,
and the model outputs `rest / fist / open-palm / pinch`.

When `--output-dir` is omitted, both training commands automatically create a
directory named `{model}__{split}__{mode}__{timestamp}__seed-{seed}`. Pass
`--artifacts-root artifacts` to write new runs under the repository-level
artifact layout.

By default, recordings are written under:

```text
data/recordings\YYYY-MM-DD_HH-MM-SS\
```

Each session contains `metadata.json`, `emg.csv`, `imu.csv`, `events.csv`, and `raw_packets.bin`.

## Hardware Protocol Summary

- Serial baud rate: `921600`
- Packet length: `29` bytes
- Header: `D2 D2 D2`
- Packet type byte: `AA` for EMG, `BB` for IMU
- Sequence byte: `0x00` to `0xFF`, checked independently for EMG and IMU
- EMG: 8 channels, 250 SPS, 24-bit signed big-endian values in microvolts
- IMU: 104 SPS, six signed int16 values decoded as gyro and accelerometer samples
