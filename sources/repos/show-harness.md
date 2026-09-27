# Show-Harness GitHub 仓库

> 来源归档（ingest）

- **项目名称：** Show-Harness
- **GitHub 地址：** <https://github.com/showlab/Show-Harness>
- **项目页：** <https://showlab.github.io/Show-Harness/>
- **论文：** <https://arxiv.org/abs/2609.10522>
- **HF 模型：** <https://huggingface.co/showlab/Show-Harness-VLMs>
- **HF 数据：** <https://huggingface.co/datasets/showlab/Show-Harness-Data>
- **核心功能：** Embodied Harness — VLM 在离散语义动作单元上闭环推理，本体解释器确定性落地；支持 frontier 零样本与 GUMI 采集 + LoRA 微调小模型。
- **入库日期：** 2026-09-27

## 仓库结构（README 对齐）

| 路径 | 作用 |
|------|------|
| `core/` | 交互主循环、配置分层、日志、共享动作词表、provider 无关 VLM 客户端（`core/vlm/`） |
| `plugins/` | Harness 插件（感知/规划/历史等阶段，配置块开关） |
| `interpreters/` | 本体解释器：Franka 阻抗、AgileX Piper 关节流、ManiSkill / Isaac Lab 仿真 |
| `gumi/` | GUMI 浏览器遥操作与 agent 操作员；每步记录 (observation, action) 训练对 |
| `configs/` | 分层配置：默认 + 站点身份 + 可选 overlay |
| `prompts/` | 零样本 controller prompt 与微调 checkpoint 的 prompt 契约 |
| `scripts/` | `setup.sh`、`run_real.py`、`run_real_mvtoken.py`、`check_setup.py`、`serve_vlm.sh` 等 |
| `train/` | LLaMA-Factory 微调：数据转换、数据集注册、LoRA 配置 |
| `models/` | Chat 模板、下载的 adapter、HF cache |
| `docs/` | Franka / Piper / 仿真 / 微调模式 runbook |

## 关键复现路径

1. **环境：** `bash scripts/setup.sh base`（真机加 `base --real`；本地 serve 另建 `.venv-vllm` + `setup.sh serve`）。
2. **GUMI 采集（仿真）：** `.venv/bin/python gumi/collect_rollouts_web.py data/rollouts_demo --sim` → 浏览器 `localhost:8600`。
3. **零样本真机：** 配置 `configs/site/franka.yaml` + `configs/secrets.env` → `python scripts/check_setup.py --robot-config configs/robot_franka.yaml` → `python scripts/run_real.py --robot-config configs/robot_franka.yaml`。
4. **微调模式：** 下载 adapter（`scripts/model/download_vlm_model.sh`）→ `scripts/serve_vlm.sh` → `python scripts/run_real_mvtoken.py --robot-config configs/robot_franka_ft.yaml`。
5. **训练：** 见 `train/README.md`（独立 venv，对接上游 LLaMA-Factory）。

## 发布物

- **Show-Harness-VLMs：** 五个 real LoRA（Qwen3.5 0.8B/2B/4B/9B、Gemma4 E4B）+ `qwen3_5_2b_sim` 仿真策略。
- **Show-Harness-Data：** 真机 Franka/Piper rollout + RoboLab / ManiSkill 演示。

## 关联 Wiki 页面

- [Show-Harness 论文实体](../../wiki/entities/paper-show-harness.md)
- [VLA 方法](../../wiki/methods/vla.md)
- [Manipulation 任务](../../wiki/tasks/manipulation.md)
