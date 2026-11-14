# Hy-Embodied-0.5-VLA 原文核对 / From VLA Models to a Real-World Robot Learning Stack

- **资料类型：** paper
- **arXiv：** 2606.14409
- **原文：** https://arxiv.org/html/2606.14409
- **官网：** https://tairos.tencent.com/openSourceModels/hy-embodied-0.5-vla
- **代码：** https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA
- **公开权重：** https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI
- **公开数据：** https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data
- **机构：** 腾讯 Robotics X、Tencent HY Team
- **核对范围：** 2026-11-14 核对 arXiv HTML 的 §1–8 及附录 A–D（模型、UMI 预训练、监督后训练、FlowPRO、部署、RoboTwin/真机实验、数据/动作坐标转换、超参数）；通读官方 README 公开资源、模型/数据/训练/评测/部署部分，静态检查 `scripts/quick_start.py` 和 `scripts/eval_robotwin_test.sh`。
- **通读：** 已完成

## 方法与口径

- 底座是 Hy-Embodied-0.5 的约 4B MoT；扩展双塔 flow-matching action expert，紧凑视觉记忆编码和与本体关节结构解耦的 **delta-chunk EEF** 动作。
- **UMI 数据：** 论文训练数据超过 10,000 小时；GitHub News 2026-06-15 明确向社会公开的是 2,000+ 小时，二者不是同一数。
- **训练：** UMI continued pretrain → target SFT（Track A 目标机器人数据、Track B UMI-only 跨本体）→ FlowPRO 失败干预/纠正数据后训练。FlowPRO 采用基于配对偏好的优化设计而非普通 PPO 奖励 critic。
- **异步部署：** 推理线程填充动作缓冲，控制线程按 servo 周期消费；通过 latency-aware Bézier chunk stitching 平滑前后动作序列。需要区分策略推理频率与控制频率。
- **RoboTwin 2.0：** 50 tasks，每任务 100 rollouts；Clean 90.9%、Randomized 90.1%，来自论文作者实验表，而非本仓库复现结果。
- **局限：** Track B 映射需针对具体臂长/姿态限制设计本体适配；硬件坐标系与多机时延不可忽略；部分 FlowPRO 资料处于审稿阶段，完整 RL 路径须单独验证。

## 源码阅读依据

- 官方 `python scripts/quick_start.py`：`snapshot_download` → `HyVLAConfig.from_pretrained` → `HyVLA.from_pretrained` → `forward_evaluate` 输出动作。
- 官方 RoboTwin 6 任务回归脚本 `scripts/eval_robotwin_test.sh`：创建 `robotwin_eval` adapter symlink → RoboTwin `script/eval_policy.py`，与真实机器人部署不是同一条入口。
- `scripts/train_umi_vlm.sh`、`scripts/train_robotwin_umi.sh` 为官方 README 给出的训练入口；未在本环境运行。

## 对 wiki 的映射

- [Hy-Embodied-0.5-VLA 唯一论文详情](../../wiki/entities/paper-hy-embodied-05-vla.md)
