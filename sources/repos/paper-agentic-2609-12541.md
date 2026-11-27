# agent-as-policy-2026/agent-as-policy — Agent as Policy 官方仓库档案

- **GitHub 地址：** https://github.com/agent-as-policy-2026/agent-as-policy
- **官方项目页：** https://agent-as-policy-2026.github.io/
- **默认分支：** `main`
- **开源协议：** 本仓代码采用 Apache-2.0（`LICENSE`）；配套 Hugging Face 数据集 [`Agent-as-Policy/agent-as-policy`](https://huggingface.co/datasets/Agent-as-Policy/agent-as-policy) 与 [`Agent-as-Policy/agent-as-policy-robodojo`](https://huggingface.co/datasets/Agent-as-Policy/agent-as-policy-robodojo) 采用 CC BY 4.0；内置第三方连接器 `third_party/graph-as-policy/` 为 Apache-2.0，`third_party/i2rt/` 补丁保持 MIT
- **关联论文：** *Agent as Policy for Robotic Manipulation*（[arXiv:2609.12541](https://arxiv.org/abs/2609.12541)）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 目录文件清单核查；未连接物理 I2RT YAM 机械臂或安装 RoboDojo 仿真器进行实机/仿真运行）。

## 核心入口与目录结构（已核实）

- **真机实验框架（`agp/`，`README.md` 与 `files_sample` 已核实）：**
  - `agp/start_bridges.sh`：按单臂或双臂启停硬件桥进程（支持 `--source fake` 无硬件模式）。
  - `agp/server_real.py` 与 `agp/robot_client.py`：真机会话服务端与智能体调用的 Python CLI 客户端，对应论文 §3.2 接口契约（`agp/README_interface_real.md`、`agp/README_interface_bare.md`、`agp/README_interface_dual.md`、`agp/README_interface_programs.md`）。
  - `agp/run_trial.sh`：受监督的单次试验启动脚本，先拍摄初始态照片并等待人工回车确认，支持 `--effort`、`--model`、`--bare`、`--right`、`--dual`、`--knowledge task|ckpt|mx` 与 `PAPER_DRYRUN=1` 干跑。
  - `agp/capture_top.sh`、`agp/consolidate_task_knowledge.sh`：顶视相机采图与跨试验经验归整脚本。
  - `agp/PROMPT_*.md`：各任务（`assembly`、`stack_blocks`、`pyramid`、`dice`、`throw_left`、`towel`、`towel2`、`singleinsert`、`twoinsert` 等）的标准与精简（`_bare`）任务定义提示词。
  - `agp/mx/`：`__init__.py`、`carry.py`、`cycle_dataset.py`、`experience.py`、`prompt.py`，管理经验积累、携带与循环数据集脚本。
  - `agp/config/`：`top_calib.json`、`right_base_in_left_base.json` 等顶视及双臂手眼/基座标定配置。
  - `agp/knowledge/` 与 `agp/paper_runs/`：跨试验保留的 `RIG_NOTES.md`、`playbooks/*.md`（如位姿回滚、倾斜抓取、偏航增量修正、分阶段插装等操作手册）、工具模块（`geometry.py`、`grip_capture.py`、`pose_and_capture.py`、`tilted_pose_capture.py`、`measure.py`）以及论文各批次 `results.csv`、`table1_metrics.csv`、`table2_metrics.csv`。
- **硬件安全桥与标定（`hardware-bridge/`、`calib/`）：**
  - `hardware-bridge/`：独立 `uv` 工程（`agp_yam_bridge` 包），独占机械臂 CAN 与相机硬件句柄，提供 `agp-yam-preflight`、`agp-yam-bridge`、`agp-yam-camera-acceptance` 三个控制台命令。
  - `calib/`：手眼与顶视相机标定运行手册（`calib/README_runbook_zh.md`）。
- **第三方依赖与基准对比模块（`third_party/`）：**
  - `third_party/i2rt/`：指向固定上游 commit 的 I2RT 机械臂 SDK 补丁（`third_party/i2rt/UPSTREAM.md` 给出拉取与打补丁步骤，构建环境前需克隆到 `<repo>/i2rt`）。
  - `third_party/graph-as-policy/`：用于基线对比的固定上游 commit 连接器（`gap`、`gap_core`）。
- **RoboDojo 仿真适配器与结果（`robodojo/`）：**
  - `robodojo/AGP/`：面向 RoboDojo / XPolicyLab 仿真基准（双 ARX X5 机械臂，仅接收官方 RGB 与本体感知，经 MCP 工具由 `gpt-6-astra` 驱动）的独立策略适配器（对应 XPolicyLab#182，不依赖上述真机桥代码）。
  - `robodojo/results/seed0/_summary.md`：全部 54 个配置共 2,100 回合的 seed 0 汇总结果（平均成功率 56.13%，平均得分 63.63）。

## 复现与运行依赖边界

- **真机环境：** 需先按 `third_party/i2rt/UPSTREAM.md` 在 `<repo>/i2rt` 检出并打补丁，再在 `hardware-bridge/` 中运行 `uv sync --locked`；无实机时可使用 `agp/start_bridges.sh` 的 `--source fake` 模式及 `PAPER_DRYRUN=1` 校验配置。大体积目标集、人类演示视频与会话录像需从 Hugging Face 数据集拉取。
- **RoboDojo 仿真环境：** 需要 Ubuntu 22.04 x86_64、带 RT 核心且显存 $\ge 16$ GB 的 NVIDIA GPU（非 A100/H100）、NVIDIA 驱动 570 或 580、conda 以及具备 `gpt-6-astra` 访问权限的 Codex CLI（固定版本 `0.160.1`）或 OpenAI API Key。
- **安全与权限边界：** 仓库未在本次核查中实机运行；开源权重（Qwen3.6 SFT / Qwen3.8-Flash-Next）的 Slime 训练脚本未在当前 `files_sample` 清单中直接列出，论文 Appendix F 给出了训练超参数。

## 与 wiki 的映射

- 对应实体页：[Agent as Policy for Robotic Manipulation](../../wiki/entities/paper-agentic-2609-12541.md)
