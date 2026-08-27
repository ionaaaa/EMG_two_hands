# EMG Two Hands

面向双手 8 通道 EMG 手环的项目：包含实时桌面端（串口接收、可视化、采集、录制与手势识别）以及接收手势事件的网页音乐游戏。

## 目录

```text
.
├── apps/
│   ├── desktop/                 # Python 桌面端，可安装包 emg_live_marker
│   └── web-game/                # 网页游戏和轻量演示 API
├── data/
│   ├── datasets/                # 原始训练数据（不提交 Git）
│   ├── recordings/              # 桌面端录制数据（不提交 Git）
│   ├── README.md
│   └── migration-manifest.csv   # 已迁移数据的校验清单
├── artifacts/                   # 产物说明与历史产物索引；实际大文件不提交 Git
│   ├── README.md
│   └── artifact_manifest.csv
├── docs/
├── third_party/                 # 外部工程，例如 EffiE（不提交 Git）
└── README.md
```

桌面端采用 `src/` 布局：包代码位于 `apps/desktop/src/emg_live_marker/`，命令入口位于 `apps/desktop/scripts/`。请不要再依赖旧的嵌套目录或 `sys.path` 修改。

## 环境与安装

项目统一使用已有 Conda 环境 **BN5213**，当前验证的解释器为 Python **3.10.19**：

```bash
PYTHON=/Applications/anaconda3/envs/BN5213/bin/python
$PYTHON --version

# 依赖已在 BN5213 中安装时使用；以可编辑模式安装桌面端包
$PYTHON -m pip install --no-deps -e apps/desktop
```

依赖声明见 [`apps/desktop/pyproject.toml`](</Volumes/UBUNTU-SERV/EMG_two hands/apps/desktop/pyproject.toml>)：PySide6、pyqtgraph、pyserial、NumPy、SciPy、PyTorch；开发依赖为 pytest 和 ruff。若 BN5213 缺少依赖，请先确认后再安装，不要使用项目内旧的 `venv/`（该目录已移除）。

下面所有 Python 命令都假设 `PYTHON` 已按上方设置，并从仓库根目录执行；可编辑安装完成后不依赖当前工作目录。

## 启动桌面端

```bash
# 无硬件：模拟 EMG 数据
$PYTHON -m emg_live_marker --simulate

# 实际设备：替换为实际串口名
$PYTHON -m emg_live_marker --port /dev/cu.usbserial-XXXX --baudrate 921600

# Windows 示例
# $PYTHON -m emg_live_marker --port COM4 --baudrate 921600
```

不带参数也会启动模拟数据源。桌面端启动本地 SSE 服务 `http://127.0.0.1:8766/events`，向网页游戏发送左右手手势事件。

桌面窗口中的 **Start Recording** 会将录制写入 `data/recordings/YYYY-MM-DD_HH-MM-SS/`；**Start Collection** 会将采集写入 `data/datasets/<subject_id>/<session_id>/`。一个会话通常包含 `metadata.json`、`emg.csv`、`imu.csv`、`events.csv` 和 `raw_packets.bin`。

## 启动网页游戏

建议经由本地 HTTP 服务启动，而不是直接双击 HTML：

```bash
$PYTHON -m http.server 8000 --directory apps/web-game
```

浏览器访问 <http://127.0.0.1:8000>。网页会优先订阅桌面端的 8766 SSE 服务；当桌面端未启动时，可用轻量演示桥接：

```bash
# 终端 A：启动演示 API
$PYTHON apps/web-game/emg_api.py --host 127.0.0.1 --port 8765

# 终端 B：发送一条模拟手势
$PYTHON apps/web-game/send_demo_gesture.py fist 0.93 --hand left
```

## 训练、微调与评估

所有训练命令可使用 `--dataset-root`、`--recordings-root`、`--artifacts-root` 和 `--paths-config` 覆盖默认路径。

```bash
# 常规手势分类训练；默认自动生成 run ID
$PYTHON -m emg_live_marker.cli.train_gesture_classifier \
  --dataset-root data/datasets \
  --device auto

# 面向同日校准数据的训练预设
$PYTHON -m emg_live_marker.cli.train_gesture_classifier \
  --dataset-root data/datasets \
  --preset calibration \
  --device auto

# EffiE 微调：需要自行提供外部 EffiE 工程和 checkpoint
$PYTHON -m emg_live_marker.cli.finetune_effie_gesture \
  --dataset-root data/datasets \
  --effie-root third_party/EffiE \
  --checkpoint-path third_party/EffiE/checkpoints/<checkpoint_file> \
  --mode freeze_backbone \
  --device auto

# 用录制会话回放评估实时平滑；替换为实际模型路径
$PYTHON -m emg_live_marker.cli.evaluate_realtime_smoothing \
  --dataset-root data/datasets \
  --model-path artifacts/models/<run_id>/gesture_classifier.ts \
  --output-dir artifacts/reports/<run_id> \
  --session session_001
```

也可以使用安装后的控制台命令，例如 `emg-train-gesture-classifier`、`emg-finetune-effie-gesture` 和 `emg-evaluate-realtime-smoothing`。

未传 `--output-dir` 时，训练与 EffiE 微调会生成：

```text
{model}__{split}__{mode}__{timestamp}__seed-{seed}
```

例如要将新产物直接写入根目录 `artifacts/`，加入 `--artifacts-root artifacts`。历史模型和报告暂未批量改名或迁移；参见 [`artifacts/artifact_manifest.csv`](</Volumes/UBUNTU-SERV/EMG_two hands/artifacts/artifact_manifest.csv>)。

## 数据和路径配置

默认路径由代码相对仓库根目录解析，不依赖“在哪个目录执行命令”：

| 用途 | 默认位置 |
| --- | --- |
| 训练数据 | `data/datasets` |
| 录制数据 | `data/recordings` |
| 模型与报告默认根 | `apps/desktop` |

路径优先级为：命令行参数 > JSON 配置文件 > 环境变量 > 默认值。可在仓库根目录创建本机私有的 `.emg-paths.json`（已被 Git 忽略）：

```json
{
  "dataset_root": "data/datasets",
  "recordings_root": "data/recordings",
  "artifacts_root": "artifacts"
}
```

对应环境变量为 `EMG_DATASET_ROOT`、`EMG_RECORDINGS_ROOT`、`EMG_ARTIFACTS_ROOT`；通过 `EMG_PATHS_CONFIG` 或 `--paths-config` 可选用其他配置文件。数据迁移过程与完整性校验见 [`data/migration-manifest.csv`](</Volumes/UBUNTU-SERV/EMG_two hands/data/migration-manifest.csv>)。

## 已验证基线（2026-08-27）

在 BN5213 / Python 3.10.19 下已确认：

- `pip install --no-deps -e apps/desktop` 成功；
- 无显示器模式可创建并关闭桌面主窗口，`MainWindow(simulate=False)` 已成功初始化；
- 路径配置、运行命名和训练产物保存的针对性测试通过：6 passed；
- 9 个数据集会话可由已安装包从仓库外的当前目录发现；首个会话 EMG 数组形状为 `(96869, 8)`；
- 当前完整测试套件尚未作为一次全量、无告警的验收重新跑完；继续迁移或改动前，应先运行下列命令。

```bash
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 \
  $PYTHON -m pytest -q apps/desktop/tests
```

最小模拟数据 smoke test（不需要硬件）：

```bash
# 终端 A
$PYTHON apps/web-game/emg_api.py --host 127.0.0.1 --port 8765

# 终端 B；预期 HTTP 200 与 ok: true
curl --fail --silent --show-error \
  -X POST http://127.0.0.1:8765/emg \
  -H 'Content-Type: application/json' \
  --data '{"samples":[[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]],"hand":"left"}'
```

## Git 约定

`.gitignore` 只会阻止未跟踪文件进入 Git，不会自动移除已经被跟踪的文件。原始数据、录制数据、模型、报告和常见生成物均应保留在本地或专用存储中，不能当作普通 Git 源码提交。

检查已跟踪的残留文件：

```bash
git ls-files | grep -E '(^|/)(venv|__pycache__|\.idea|data|artifacts)/|\.egg-info/|/\._' || true
```

当前结果不包含虚拟环境、缓存、IDE 元数据、`._*` 或 `*.egg-info`。`data/README.md`、迁移清单、`artifacts/README.md` 与产物索引是有意跟踪的元数据；原始数据和大产物仍被忽略。若确认某个已跟踪生成物不应在 Git 中，只取消跟踪并保留本地文件，例如：

```bash
git rm -r --cached -- apps/desktop/venv
```

不要对不确定的数据目录批量执行 `git rm --cached`。
