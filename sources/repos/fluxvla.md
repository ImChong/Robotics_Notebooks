# FluxVLA Engine（FluxVLA/FluxVLA）

> 来源归档

- **标题：** FluxVLA Engine — A One-Stop VLA Engineering Platform for Embodied Intelligence
- **类型：** repo / platform
- **组织：** [FluxVLA](https://github.com/FluxVLA)（逐际动力 LimX Dynamics 主导）
- **代码：** <https://github.com/FluxVLA/FluxVLA>
- **文档：** <https://fluxvla.limxdynamics.com/>（[中文](https://fluxvla.limxdynamics.com/zh/)）
- **产品页：** <https://www.limxdynamics.com/zh/products/fluxvla>
- **论文：** <https://arxiv.org/abs/2609.17210>（Zenodo [10.5281/zenodo.20049506](https://doi.org/10.5281/zenodo.20049506)）
- **HF 权重：** <https://huggingface.co/limxdynamics/FluxVLAEngine>
- **入库日期：** 2026-09-06（国内开源全景初录）
- **复核日期：** 2026-09-28（arXiv 论文 ingest + README / 仓库结构核对）
- **一句话说明：** 配置驱动的 **VLA 全栈工程平台**：LeRobot 兼容数据、`fluxvla/` 训练/评测/推理引擎、多 backbone 微调（π0/π0.5、GR00T、Cosmos3、FastWAM、DiT4DiT、DreamZero、SmolVLA 等）、LIBERO / RoboCasa / RoboDojo 统一评测、RTC + Triton + ZMQ 远程推理与 Franka / Oli 真机 runner。

## 开源状态（README / 项目页核查 2026-09-28）

- **已开源：** 主仓 Python 包 `fluxvla/`（`datasets` / `models` / `engines` / `evaluators` / `collators` 等）；入口 `scripts/train.py`（torchrun）、`scripts/eval.sh`、`scripts/inference_real_robot.py`；`configs/` 按策略族组织；`bash scripts/install_env.sh` 提供 sim-only / real-only / full 一键环境。
- **权重与复现：** HF `limxdynamics/FluxVLAEngine` 链出 LIBERO / RoboCasa 等 checkpoint；README Performance 表与链接一一对应。
- **生态子仓：** [FluxDAgger](https://github.com/FluxVLA/FluxDAgger)（模型解耦 DAgger）、[FluxBisim](https://github.com/FluxVLA/FluxBisim)（双臂仿真基准）。
- **依赖栈：** 致谢 LeRobot、Isaac GR00T、OpenPI、OpenVLA、DreamZero、DeepSpeed、RTC 等；真机 `real-only` 需系统 ROS（Noetic 等）与机器人 SDK 集成点。
- **上游活跃：** GitHub `pushed_at` 2026-09-25（复核窗口内仍有推送）。

## 快速入口（对齐 README）

```bash
conda create -n fluxvla python=3.10 -y && conda activate fluxvla
bash scripts/install_env.sh sim-only   # 或 real-only / full

export WANDB_MODE=disabled
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --work-dir ./work_dirs/pi05_paligemma_libero_10_full_finetune

bash scripts/eval.sh [CONFIG] [CKPT_PATH]
python scripts/inference_real_robot.py --config [CONFIG] -- ckpt-path [CKPT_PATH]
```

数据工具：`tools/convert_hdf_to_lerobot.py`、`tools/arm_awbc`（ARM 奖励与 AW-BC 重加权）等。

## 对 wiki 的映射

- [FluxVLA Engine（实体）](../../wiki/entities/fluxvla-engine.md)
- [LimX COSA](../../wiki/entities/limx-cosa.md) — 闭源 OS vs 开源技能层
- [VLA 部署 12 篇地图](../../wiki/overview/vla-deploy-12-papers-technology-map.md)
- [LeRobot](../../wiki/entities/lerobot.md) — 数据格式互操作
