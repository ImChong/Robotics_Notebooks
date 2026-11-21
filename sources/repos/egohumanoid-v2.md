# EgoHumanoid-V2（OpenDriveLab）源码核查

> 来源归档（repo / 官方实现）

- **仓库：** <https://github.com/OpenDriveLab/EgoHumanoid-V2>
- **组织：** OpenDriveLab
- **默认分支：** `main`
- **可见性：** public；GitHub repo API 核查仓库存在且可公开读取
- **许可证：** Apache-2.0（根目录 `LICENSE`）
- **项目页：** <https://opendrivelab.com/EgoHumanoid-V2/>
- **论文：** <https://arxiv.org/abs/2609.37181>
- **项目页归档：** [egohumanoid-v2-opendrivelab.md](../sites/egohumanoid-v2-opendrivelab.md)
- **Wiki 主实体：** [EgoHumanoid-V2](../../wiki/entities/paper-egohumanoid-v2.md)
- **核查日期：** 2026-10-12（2026-11-21 复核至 commit `682199e`，2026-11-04）
- **核对范围：** 拉取 `OpenDriveLab/EgoHumanoid-V2` @ `682199e`（"Release policy training and inference for motion tokens and Revo2 hands"），通读 `README.md`、`data_alignment/action_alignment/src/pipeline.py`、`data_alignment/vision_alignment/src/pipeline.py`、`model_training/src/train.py` 与 `model_training/src/serve.py`。边界：未在本地 GPU/G1 真机运行。
- **通读：** 已完成

## 仓库覆盖范围

官方 README 提供 Ubuntu/Python/CUDA/SONIC/PICO/G1/Revo2 等硬件与环境配置，并完整公开人类示范采集、同步、动作对齐、视觉对齐、OpenPI π₀.₅ 微调与 SONIC 推理部署代码。已公开的重要入口如下：

| 用途 | README 入口 / 路径 | 状态 |
|------|-------------------|------|
| 人类数据采集 | `eh2-collect-human` | 已公开；网页端录制 LeRobot 格式数据 |
| 同步与质量检查 | `eh2-sync-human --root data/{task}` | 已公开；50 Hz 同步并生成质检报告 |
| 粗到细动作对齐 | `data_alignment/action_alignment`；`eh2-align --input ... --endpoints hands\|feet --output ...` | 已公开；含四项任务样例与预训练 response MLP |
| 前后动作仿真回放 | `eh2-replay --mode smpl\|token` | 已公开；结果写 `replay.mp4` |
| 视觉对齐 | `data_alignment/vision_alignment`；`eh2-visual-align` | 已公开（升级至 **SAM 3.1** + DiffuEraser + G1 手臂渲染；论文为 SAM 2） |
| VLA 策略训练 | `model_training`；`eh2-norm-stats`、`eh2-train` / `torchrun --module eh2_policy.train` | 已公开（2026-11-04 `682199e` 新增；基于 OpenPI PyTorch `pi05_base`） |
| 策略部署 | `eh2-serve`（FastAPI `/health` & `/act`）+ SONIC `run_vla_inference.py` | 已公开（2026-11-04 `682199e` 新增；43 维状态 → action_horizon × 76 动作） |

README 声明 action alignment 示例输入放在 `{task}/input.parquet`，对齐输出目录含 `data.parquet`（人体 SMPL 动作及对齐 token）和 `run.json`（逐轮指标与被选轮次）；视觉对齐在仓库中按 `STAGE_DEPENDENCIES`（`segment -> geometry & inpaint -> render`）执行并用输入与代码 sha256 记录 `run.json` 断点；策略服务校验 43 维状态与 76 维动作（64 维 SONIC token + 左右手各 6 维 Revo2 命令）。

## 开源与复现边界

- **代码链已完整开源（Apache-2.0）。** 从采数、动作/视觉对齐到 OpenPI 微调与 SONIC 服务部署均具备可运行 CLI 与模块，可直接对齐源码绘制完整运行时序图。
- **数据与外部权重仍需自备。** 仓库提供动作对齐示例、response MLP 权重与 3 项视觉对齐样例，未随仓提供论文全量 200 条/任务人类示范数据集与任务微调后策略权重；复现者需另行获取 `pi05_base`、SONIC 控制器与权重、SAM 3.1 权重及 G1/Revo2/PICO/GoPro 硬件。
- **视觉分割模型版本差异。** 论文 §3.4 使用 SAM 2，而公开仓库实现已升级为 SAM 3.1（需 Hugging Face 授权下载）。

## 上下游关系

- 项目页：<https://opendrivelab.com/EgoHumanoid-V2/>
- 论文来源：[egohumanoid_v2_arxiv_2609_37181.md](../papers/egohumanoid_v2_arxiv_2609_37181.md)
- 项目页来源：[egohumanoid-v2-opendrivelab.md](../sites/egohumanoid-v2-opendrivelab.md)
- Wiki 实体：[paper-egohumanoid-v2.md](../../wiki/entities/paper-egohumanoid-v2.md)
