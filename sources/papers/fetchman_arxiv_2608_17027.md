# FetchMan（arXiv:2608.17027）

> 来源归档（ingest）

- **标题：** FetchMan: Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences
- **类型：** paper / humanoid / loco-manipulation / sim2real / visual-rl / flow-matching / grpo
- **arXiv：** <https://arxiv.org/abs/2608.17027>
- **项目页：** <https://orayyan.com/fetchman>
- **作者：** Omar Rayyan、Zhi Li、Max Argus、Yuxin Jiang、Chang Yu、Chenfanfu Jiang、Yuchen Cui
- **机构：** 加州大学洛杉矶分校（UCLA）；艾伦人工智能研究所（Allen Institute for AI）；华盛顿大学（University of Washington）
- **入库日期：** 2026-08-20
- **一句话说明：** 在 MolmoSpaces 上生成 15 万场景脚本演示 → DINOv3+DiT BC → Flow-GRPO 稀疏奖励 refinement → Unitree G1 零样本 loco-manip。

## 开源状态（步骤 2.5）

- **项目页（2026-08-20）：** 含摘要、架构、消融视频与 BibTeX；**未列 GitHub / Hugging Face / 权重下载**。
- **论文：** 宣称 release **FetchMan-Bench**（固定 held-out 场景与评分），但未给代码 URL。
- **结论：** **确认未开源**（训练代码与 checkpoint 截至入库日不可获取）；基准「将发布」待跟进。

## 摘录 1：问题与管线（§1、§4–5）

- **设定：** Unitree G1 + Dex1-1；15 维全身命令 @10 Hz；SONIC 低层 WBC @50 Hz 跟踪 base，上身 PD @200 Hz。
- **观测：** 头 fisheye + 腕 RGB（224×384）+ 本体（高度/roll/pitch/上身关节/夹爪）。
- **数据：** MolmoSpaces 程序化室内场景；~150k 场景、~50k 物体；脚本特权控制器 \(\pi_{\text{ctrl}}\) 生成 ~650 h 演示（单 L40S ~40 GPU-h）。
- **痛点：** 脚本演示含 **不可观测相位边界**（导航↔reach↔manip），BC 天花板低；更多数据无法突破。

**对 wiki 的映射：** 与 [VIRAL](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md) 对照：同为 G1 视觉 loco-manip，FetchMan 强调 **环境泛化 + Flow-GRPO 破 BC 顶**。

## 摘录 2：策略与 Flow-GRPO（§5）

- **BC：** 冻结 DINOv3 ViT-B/16 patch token + 本体 MLP state token → DiT flow-matching 预测 H=16 chunk、执行前 8；**delta 动作** 重参数化。
- **RL：** Flow-GRPO：5 步 Gaussian SDE 给逐步 log-lik；64 组×8 episode 同 reset；稀疏 grasp+lift 奖励；PPO clip + 冻结 BC KL。
- **增益：** 单物体 sim loco-manip 67%→83%；真机 56.7%→73.3%；纯 manip 提升小（72.7%→77.2%）——瓶颈在 **行走/ reposition**。

**对 wiki 的映射：** 链接 [loco-manipulation](../../wiki/tasks/loco-manipulation.md)、Flow-GRPO 文献。

## 摘录 3：评测与消融（§6）

- **FetchMan-Bench：** manipulation SR（ grasp standoff 起）与 loco-manipulation SR（远处起步）；sim 100 held-out init，真机 zero-shot。
- **关键消融：** DINOv3 + delta 动作 **缺一不可**；换 SigLIP 或 absolute 动作真机 loco-manip **≈0%**。
- **多物体：** dino.txt 文本条件；350k 演示；BC 40% → Flow-GRPO 62% sim loco-manip。

**对 wiki 的映射：** 结论强调「BC 顶 + RL 补 walking」与 sim2real 两要素。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-fetchman.md`**；`sources/sites/fetchman-orayyan.md`。
- 更新 [loco-manipulation](../../wiki/tasks/loco-manipulation.md) 视觉 sim 路线索引。
