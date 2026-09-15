# ORCS（lok-i/orcs）— 原始资料归档

- **来源：** <https://github.com/lok-i/orcs>
- **类型：** repo
- **机构：** 南加州大学（USC）— Lokesh Krishna 等（ViBe 论文团队）
- **归档日期：** 2026-09-15
- **全称：** **O**racle **R**obot **C**ontrol **S**ynthesis（Optimize, Retarget, Control Suite）
- **许可证：** ORCS 代码与文档 **BSD-3-Clause**；公开发布 checkpoint 含 **SONIC** 权重，受 **NVIDIA Open Model License** 约束
- **权重：** <https://huggingface.co/lkrajan/orcs>（`v0.1.0`，SHA-256 见 `src/orcs/release.json`）

## 一句话说明

**ORCS** 是面向 **任务后训练（post-training）特权人形全身控制器** 的工具包：在 **冻结 SONIC** 运动跟踪基座上挂 **零初始化 LoRA 适配器**，用 **不对称 actor-critic（特权 critic + 任务条件 augmentation）** 在 mjlab 中 PPO 训练；附带 **SMPL 运动学重定向** 与 **PerLoco / UOLM / Dodge** 三类任务脚手架。为 [ViBe](https://arxiv.org/abs/2609.09918) 研究线配套工程栈。

## 开源核查（步骤 2.5）

| 项 | 结论（截至 2026-09-15） |
|----|-------------------------|
| **代码** | **已开源** — <https://github.com/lok-i/orcs>（BSD-3-Clause） |
| **公开 checkpoint** | **已发布** — HF `lkrajan/orcs`：`Orcs-Dodge-AdaptSonic`、`Orcs-PerLoco-{Grail,OmRe}-AdaptSonic`、`Orcs-Uolm-AdaptSonic` |
| **SONIC 基座权重** | 随 AdaptSonic 任务依赖；公开发布 ckpt **含 SONIC 权重**，NVIDIA Open Model License |
| **ViBe 视觉后训练** | **未随 ORCS 发布** — README 称 ORCS 为 ViBe 开发产物；**student 蒸馏（oracle→可部署）在 roadmap** |
| **外部数据** | PerLoco 需 `perceptive_locomotion.sh`（GRAIL / OmniRetarget）；UOLM 需重建人体-物体运动；SMPL-X **需单独许可下载** |

## 为什么值得保留

- **「特权先训、再蒸馏」工程样板：** `docs/ethos.md` 明确 oracle（全 sim 状态 + 特权 critic）与 student（机载感知）分阶段；当前 shipped 为 **AdaptSonic + LoRA**，蒸馏为下一里程碑。
- **SONIC 生态补全：** 在 [GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md) 官方 SONIC 训练栈之外，提供 **任务级 LoRA 后训练** 与 **play/train CLI**。
- **可立即试玩：** `play <Task> --agent release --viewer native` 拉 HF 校验 checkpoint；Dodge 无外部运动数据集即可跑。

## 核心能力

| 能力 | 说明 |
|------|------|
| **LoRA PEFT on SONIC** | `SonicWithAdapterModel`；`--agent initial` 应 bit-exact 复现冻结基座 |
| **Kinodynamic SMPL retargeting** | `orcs-pseudo-retarget` 生成 seed state；PerLoco/UOLM 支持 `robot` / `smpl` command space |
| **任务族** | **UOLM**（Uni-Object Loco-Manipulation）、**PerLoco**（地形高度扫描感知 locomotion）、**Dodge**（全身躲球） |
| **仿真栈** | mjlab 1.4 + MuJoCo 3.8；`sync_dependencies.sh` 钉死 `mocke` / `rsl_rl` 可编辑 fork |

## 公开 release checkpoint（v0.1.0）

| 任务 ID | 训练 iteration | 外部数据依赖 |
|---------|----------------|--------------|
| `Orcs-Dodge-AdaptSonic` | 7,500 | 无 |
| `Orcs-PerLoco-Grail-AdaptSonic` | 14,999 | GRAIL + SMPL-X（`perceptive_locomotion.sh`） |
| `Orcs-PerLoco-OmRe-AdaptSonic` | 10,500 | OmniRetarget + GRAIL |
| `Orcs-Uolm-AdaptSonic` | 19,999 | UOLM 重建人体-物体运动 |

缓存默认：`~/.cache/orcs/releases`（`ORCS_RELEASE_ROOT` 可改）。

## 关键复现路径

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e .
bash scripts/setup/sync_dependencies.sh   # 必须最后执行，钉死 mocke/rsl_rl
bash scripts/setup/download_released_models.sh   # 可选：预拉 HF release
play Orcs-Dodge-AdaptSonic --agent release --viewer native
train Orcs-Uolm-AdaptSonic --env.scene.num-envs 4096
```

PerLoco：额外 `uv pip install -e ".[perloco]"` + `bash scripts/setup/perceptive_locomotion.sh`。

## 对 wiki 的映射

- [orcs](../../wiki/entities/orcs.md) — 本仓升格实体页
- [paper-vibe](../../wiki/entities/paper-vibe.md) — 研究出处；视觉后训练仍待发布
- [sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md) — 冻结基座
- [privileged-training](../../wiki/concepts/privileged-training.md) — 不对称 actor-critic 范式
- [gr00t-wholebodycontrol](../../wiki/entities/gr00t-wholebodycontrol.md) — 官方 SONIC 训练/部署对照

## 引用

```bibtex
@misc{krishna2026vibe,
  title={ViBe: Visual Behavior Adaptation for Perceptive Humanoid Whole-Body Control},
  author={Lokesh Krishna and Sarvesh Venkatesan and An Zhang and Quan Nguyen},
  year={2026},
  eprint={2609.09918},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.09918},
}
```
