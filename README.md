# EMG Two Hands

两只 8 通道 EMG 手环驱动的桌面端采集/识别程序与网页音乐游戏。

本文件是目录迁移前后的**运行与验证基线**。迁移任何路径后，都应从本文件的“验证基线”开始重新检查；不要以旧目录中的虚拟环境或训练产物是否存在作为成功标准。

## 当前代码位置

- Python 桌面端：`apps/desktop/`
- 网页游戏与轻量调试 API：`apps/web-game/`
- 桌面端会在 `http://127.0.0.1:8766/events` 提供手势 SSE。
- 网页游戏优先连接 8766；连接失败时回退到轻量 API 的 `http://127.0.0.1:8765/events`。

## Python 与依赖安装

项目声明的最低 Python 版本是 **3.10**（见 `apps/desktop/pyproject.toml`）。此项目统一使用 Conda 环境 **BN5213**：

```bash
PYTHON=/Applications/anaconda3/envs/BN5213/bin/python
$PYTHON --version
$PYTHON -m pip install -e apps/desktop
```

此环境当前为 Python **3.10.19**。本次检查发现其缺少 `PySide6`、`pyqtgraph`、`pyserial` 和 `ruff`；这些依赖**没有被自动安装**。补齐依赖前，桌面 GUI、完整测试与可编辑安装的端到端验证都无法完成。

后续所有命令均在设置了上述 `PYTHON` 变量的同一终端中执行。应用采用 `src/` 布局；完成可编辑安装后，运行时与测试应从已安装的 `emg_live_marker` 包导入代码，而不是依赖工程目录在 `sys.path` 中。

如需单独创建本机环境，仍可在明确授权后使用以下方式；默认不替代 BN5213：

```bash
cd apps/desktop
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell（替代上一行）
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

仓库中的 `apps/desktop/venv/` 是 Windows 可执行文件格式，不能在 macOS 上复用；请创建上述 `.venv/`。两种环境目录都被 `.gitignore` 排除。

依赖包括 PySide6、pyqtgraph、pyserial、NumPy、SciPy、PyTorch；开发依赖包括 pytest 和 ruff。

## 启动方式

### 桌面端

```bash
cd apps/desktop

# 不接硬件：使用模拟数据启动
$PYTHON -m emg_live_marker --simulate

# 真实串口：示例端口名请替换为实际设备
$PYTHON -m emg_live_marker --port /dev/cu.usbserial-XXXX --baudrate 921600

# Windows 示例
# $PYTHON -m emg_live_marker --port COM4 --baudrate 921600
```

不带参数启动时，当前实现也会进入模拟模式。桌面端打开后会启动本地 SSE 服务（8766），供网页游戏接收左右手手势。

### Web 游戏

推荐通过静态 HTTP 服务打开网页，而不是直接双击 HTML 文件：

```bash
# 在仓库根目录执行
python -m http.server 8000 --directory apps/web-game
```

然后浏览器访问 `http://127.0.0.1:8000`。如桌面端正在运行，网页会连接其 8766 SSE 服务。

没有桌面端或硬件时，可启动轻量演示桥接：

```bash
# 终端 A：仓库根目录
python apps/web-game/emg_api.py --host 127.0.0.1 --port 8765

# 终端 B：向网页发送一条模拟手势
python apps/web-game/send_demo_gesture.py fist 0.93 --hand left
```

### 录制与采集

当前没有独立的录制命令行脚本。启动桌面端后，在窗口中使用：

- **Start Recording**：写入录制根目录下的 `YYYY-MM-DD_HH-MM-SS/`；
- **Start Collection**：写入数据集根目录下的 `<subject_id>/<session_id>/`，用于训练。

每个会话包含 `metadata.json`、`emg.csv`、`imu.csv`、`events.csv` 与 `raw_packets.bin`。这些都是本地数据产物，当前不应提交到 Git。

## 训练与离线评估

以下命令均从 Python 工程根目录执行：

```bash
cd apps/desktop

# 常规训练；输出目录请使用新的、语义明确的实验名
$PYTHON scripts/train_gesture_classifier.py \
  --dataset-root apps/desktop/emg_live_marker/dataset \
  --output-dir apps/desktop/models/<run_id> \
  --device auto

# EffiE 微调；需要自行提供外部 EffiE 工程和 checkpoint
$PYTHON scripts/finetune_effie_gesture.py \
  --dataset-root apps/desktop/emg_live_marker/dataset \
  --effie-root ../../third_party/EffiE \
  --checkpoint-path ../../third_party/EffiE/checkpoints/<checkpoint_file> \
  --output-dir apps/desktop/models/<run_id> \
  --device auto

# 回放评估实时平滑效果
$PYTHON scripts/evaluate_realtime_smoothing.py \
  --dataset-root apps/desktop/emg_live_marker/dataset \
  --model-path apps/desktop/models/<run_id>/gesture_classifier.ts \
  --output-dir apps/desktop/reports/<run_id> \
  --session session_001
```

## 数据与产物路径

路径不依赖执行命令时的当前目录。在实际搬迁前，默认仍指向旧位置：

- dataset：`apps/desktop/emg_live_marker/dataset`
- recordings：`apps/desktop/recordings`
- artifacts：`apps/desktop` （其中 `models/` 与 `reports/` 为模型和报告目录）

优先级从高到低是：命令行 `--dataset-root` / `--recordings-root` / `--artifacts-root`、路径配置文件、环境变量、上述默认值。本机可在仓库根目录创建未跟踪的 `.emg-paths.json`：

```json
{
  "dataset_root": "data/datasets",
  "recordings_root": "data/recordings",
  "artifacts_root": "artifacts"
}
```

对应的环境变量是 `EMG_DATASET_ROOT`、`EMG_RECORDINGS_ROOT`、`EMG_ARTIFACTS_ROOT`；用 `EMG_PATHS_CONFIG` 或 `--paths-config` 可选择其他 JSON 配置文件。模型与报告分别位于 `artifacts_root/models` 和 `artifacts_root/reports`。

## 验证基线

### 已通过的自动测试

在 2026-08-27、BN5213 Python 3.10.19 下，以下临时结构验证命令通过：

```bash
cd apps/desktop
PYTHONPATH=src $PYTHON -m pytest -q \
  tests/test_processing.py \
  tests/test_protocol.py \
  tests/test_realtime_smoothing_eval.py \
  tests/test_ring_buffer.py \
  tests/test_stream_processor.py
```

结果：**38 passed in 14.04s**。`PYTHONPATH=src` 仅用于本次未安装依赖时验证 `src/` 导入结构；它不是日常运行方式。

补齐依赖并完成可编辑安装后，日常与 CI 验证统一使用：

```bash
cd apps/desktop
QT_QPA_PLATFORM=offscreen $PYTHON -m pytest -q
```

本次尚未能完整执行：当前验证环境缺少 `PySide6`、`pyqtgraph` 和 `pyserial`，使 5 个 GUI/解码器测试模块在收集时失败。安装本节前述依赖后，应重新执行完整命令。

### 最小模拟数据 smoke test

此测试验证轻量桥接能够接收并分类一条合成的 8 通道 EMG 样本。它不验证已训练模型，也不替代桌面 GUI 验证。

```bash
# 终端 A：仓库根目录
python apps/web-game/emg_api.py --host 127.0.0.1 --port 8765

# 终端 B
curl --fail --silent --show-error \
  -X POST http://127.0.0.1:8765/emg \
  -H 'Content-Type: application/json' \
  --data '{"samples":[[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]],"hand":"left"}'
```

本次实际返回 HTTP 200，响应为 `ok: true`，并将样本判为 `open-palm`（置信度是演示 API 的随机值）。使用 `Ctrl-C` 停止临时服务。

## Git 与迁移卫生检查

`.gitignore` 只会阻止**尚未跟踪**的文件进入 Git，不能自动停止跟踪已提交过的文件。每次迁移前后执行：

```bash
git ls-files | grep -E '(^|/)(venv|__pycache__|\.idea|data|artifacts)/|\.egg-info/|/\._' || true
```

本分支当前的结果为空：没有这些生成物被 Git 跟踪。如果未来发现明确不应进入仓库的已跟踪生成物，只取消跟踪而保留本地文件，例如：

```bash
git rm -r --cached -- apps/desktop/venv
```

不要对 `dataset/`、`recordings/` 或其他不确定的数据目录批量执行 `git rm --cached`；先确认其用途、备份和数据保留策略。
