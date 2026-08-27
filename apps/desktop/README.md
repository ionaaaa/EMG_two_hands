# emg_live_marker

`emg_live_marker` is a Python desktop app for real-time display of 8-channel Bluetooth transparent-serial EMG wristband data, with event marking, simulation, serial connection, and session recording.

It supports simulated data for development without hardware and real serial input through pyserial.

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Test

```powershell
pytest
```

## Run

```powershell
python -m emg_live_marker --simulate
python -m emg_live_marker --port COM4
```

No hardware is required for the simulator:

```powershell
python -m emg_live_marker --simulate
```

Running without arguments also starts the simulator:

```powershell
python -m emg_live_marker
```

Real serial mode on Windows:

```powershell
python -m emg_live_marker --port COM4 --baudrate 921600
```

## Calibration Game Model

For live game demos, cross-session accuracy can be low because bracelet placement
changes between sessions. After wearing the bracelet, collect a small calibration
set for the three gestures, then train a same-day game model:

```powershell
python scripts/train_gesture_classifier.py ^
  --dataset-root dataset ^
  --output-dir models/calibration_game_model ^
  --preset calibration ^
  --device auto
```

The calibration preset trains the `emg2pose_tcn` classifier with balanced rest
windows, balanced sampling/loss, best-checkpoint saving, and TorchScript export.
The Game Decoder will prefer `models/calibration_game_model/gesture_classifier.ts`
when it exists.

## EffiE Transfer Learning

`emg2pose` checkpoints are not used directly because they are 2kHz, 16-channel
pose-regression models. For a public model closer to this hardware, use the
Myo/8-channel/200Hz EffiE project:

```powershell
git clone https://github.com/MIC-Laboratory/IEEE-NER-2023-EffiE external_models\EffiE
dir external_models\EffiE\checkpoints
```

Freeze the EffiE-style backbone and train a 4-class game head:

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

If freezing the backbone is not enough, fine-tune the whole model with a lower
learning rate:

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

The exported `gesture_classifier.ts` uses EffiE-style preprocessing: raw 250Hz
EMG is resampled to 200Hz, the latest 32 samples are used as an `8x32` window,
and the model outputs `rest / fist / open-palm / pinch`.

Recordings are written under:

```text
recordings\YYYY-MM-DD_HH-MM-SS\
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
