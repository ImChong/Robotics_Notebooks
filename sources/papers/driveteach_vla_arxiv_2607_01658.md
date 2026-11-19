# DriveTeach-VLA: Teaching Vision-Language-Action Models What to See and Where to Look（arXiv:2607.01658）

> 来源归档（ingest）

- **标题：** Teaching Vision-Language-Action Models What to See and Where to Look
- **类型：** paper / vla / autonomous-driving / end-to-end-planning / navsim
- **arXiv：** <https://arxiv.org/abs/2607.01658>
- **代码：** <https://github.com/ShivaTeam/DriveTeach-VLA>
- **机构：** 北京航空航天大学（BUAA）、清华大学智能产业研究院（AIR）、滴滴、中国传媒大学
- **出处：** **ECCV 2026**（README 称 2026-06-24 接收）
- **入库日期：** 2026-09-21
- **复核日期：** 2026-11-19
- **原文：** <https://arxiv.org/html/2607.01658v1>（v1，含正文、实验、附录 A/B/C）
- **核对范围：** 论文 HTML v1 全文：摘要、引言、Related Work、Method 3.1–3.3、Experiments 4.1–4.3（Tables 1–10）、Conclusion 与 Appendix 0.A–0.C（Tables 11–14、system prompts）；交叉核对官方 README、data_engine/README.md、dvd/README.md 及已发布的核心源码与 YAML 配置。
- **通读：** 已完成
- **一句话说明：** 用 Driving-aware Vision Distillation（DVD）教 VLA「看什么」、2D Trajectory-Guided Prompts（2D-TGP）教「看哪里」，再经 TGP-guided SFT + GRPO 对齐 BEV 轨迹；NAVSIM 与 nuScenes SOTA；Apache-2.0 已开源。

## 开源状态（步骤 2.5，2026-09-21）

| 资源 | 状态 |
|------|------|
| arXiv PDF | **已发布** |
| GitHub 仓库 | **已开源** Apache-2.0 — `data_engine/`、`dvd/`、`sft/`（LLaMA-Factory 补丁） |
| 标注与数据集 | **已发布** — [Google Drive](https://drive.google.com/drive/folders/1oOz6EVfsrxGvOXvYkwxNhcgqjYP-LqSa?usp=drive_link) |
| GRPO 训练代码 | **部分外置** — 见 [Curious-VLA](https://github.com/Mashiroln/curious_vla) |
| 结论 | **已开源**（主仓训练/数据引擎 + 外站数据；RL 阶段需 Curious-VLA） |

## 核心论文摘录

### 1) 问题与动机（Abstract）

- **缺口：** 现有驾驶 VLA 过度依赖 **文本中心 VQA / CoT** 预训练，强调语言推理而非 **action-grounded planning**；表征有语义但缺 **空间依赖**，轨迹预测不稳。
- **主张：** 显式教 VLA **what to see** 与 **where to look**，再 **how to act**。
- **对 wiki 的映射：**
  - [DriveTeach-VLA 论文实体](../../wiki/entities/paper-driveteach-vla.md)
  - [VLA 方法页](../../wiki/methods/vla.md)

### 2) 三阶段 vision-guided pipeline

| 阶段 | 模块 | 作用 |
|------|------|------|
| **What to see** | **DVD**（Driving-aware Vision Distillation） | bbox 增强图像 → 自蒸馏，向视觉编码器注入 **交通场景先验** |
| **Where to look** | **2D-TGP** + TGP-guided **SFT** | 相机投影可行轨迹关键点，提供与驾驶路径对齐的 **空间 grounding** |
| **How to act** | TGP-guided **GRPO** | 强化学习进一步对齐轨迹预测与驾驶偏好 |

- **推理：** 双模型 — **TGP-Prompter** 从前视图像预测 2D-TGP → **TGP-Planner** 条件于 2D-TGP 生成 **BEV 轨迹**。
- **对 wiki 的映射：**
  - [GRPO](../../wiki/methods/grpo.md)
  - [S²-VLA](../../wiki/entities/paper-s-squared-vla.md)（同为 NAVSIM 驾驶 VLA 对照）

### 3) 实验（Abstract / README）

- **基准：** **NAVSIM**、**nuScenes** — 报告 **SOTA 级**（具体分项见原文 Table）。
- **数据：** 基于 **NAVSIM v2.0.0**；`data_engine/` 可配置 Extract → Enrich → Render 流水线（SampleIR、2D-TGP 投影、LLaMA-Factory 格式）。
- **骨干：** **Qwen2.5-VL** + LLaMA-Factory SFT；DVD 预训练在 `dvd/`。
- **对 wiki 的映射：**
  - [Autonomous Driving Core Algorithms](../../wiki/overview/autonomous-driving-core-algorithms-series.md)

### 4) 工程与复现（README）

- **Data Engine 三 pipeline：** `poutine_label`（VLM 伪标签）、`prompter`（DVD / 2D-TGP）、`planner`（CoT-SFT）。
- **训练入口：** `python data_engine/main.py --config ...`；DVD/SFT 见 `dvd/README.md`。
- **依赖：** Qwen2.5-VL、NAVSIM、Transformers、LLaMA-Factory；Prompt 设计参考 ReCogDrive、Poutine。
- **对 wiki 的映射：**
  - [sources/repos/driveteach-vla.md](../repos/driveteach-vla.md)

### 5) 局限

- **RL 代码外置：** GRPO 阶段不在主仓，需 Curious-VLA。
- **双模型推理：** Prompter + Planner 两趟前向，部署需算力与延迟预算。
- **域绑定：** 数据引擎与评测强依赖 NAVSIM / 驾驶前视设定，迁移到其他机器人操作域需重设计 TGP。

## 全文精读与数值复核（2026-11-19）

### 方法与监督边界（§3）

- **双 3B 模型：** TGP-Prompter 与 TGP-Planner 均基于 Qwen2.5-VL-3B，但优化目标不同。Prompter 在训练中学习从前视图像预测 2D-TGP；Planner 以原始图像、ego-state、语言和 2D-TGP 文本坐标为条件，输出未来 BEV 轨迹，避免两种指令混训。
- **DVD（Eq. 5–8）：** Grounding DINO 离线生成交通关键物体框，teacher ViT 吃 bbox 叠加图，student ViT 吃原图；对视觉 token 的空间分块均值做 Smooth-L1 对齐。teacher 由 student 的 EMA 更新，并联 2D-TGP 自回归交叉熵：`L_DVD = L_distill + λ_TGP × L_TGP`。论文实验使用 `λ_TGP=0.1`、`EMA=0.996`、`2×4` 分块。
- **2D-TGP（Eq. 1–4）：** 已知相机内外参，专家 BEV 轨迹投到前视像素坐标；利用车辆所在路面 `z=0` 假设，2D 点可反投影回 BEV（**并非一般三维空间可逆**）。Planner 的 SFT / GRPO 使用**真值** 2D-TGP（teacher forcing），部署则用 Prompter 的**预测**提示，不能在评估中喂真值泄漏答案。
- **GRPO（Eq. 9）：** 对预测轨迹用 PDM-Score 奖励，NC、DAC 为乘法门限，EP / TTC / Comfort 权重为 5/5/2；论文遵循 Curious-VLA 的过滤和 rollout。nuScenes 实验**没有**使用 CoT 和 GRPO，切勿把 NAVSIM 训练条件套用到 nuScenes。

### 核心结果及对照协议（§4 / Appendix）

| 指标 / 设置 | DriveTeach-VLA | 正确读法 |
|---|---:|---|
| NAVSIM navtest 单次推理 PDMS（Table 1） | **90.4** | Qwen2.5-VL-3B；同表 AutoVLA 89.1 / CuriousVLA 88.9（论文复现） |
| NAVSIM 消融：原始 Qwen 3B → VQA+CoT → DVD+CoT（Planner）→ +TGP → +GRPO（Table 5） | **84.8 → 86.4 → 87.1 → 88.2 → 90.4** | 不同阶段不可混称相同 SFT 预算；同表仅 DVD+CoT+GRPO、无 TGP 为 89.1 |
| nuScenes ST-P3 口径：3 秒内平均 L2 / 碰撞率（Table 2） | **0.30 m / 0.12%** | UniAD 口径为 **0.60 m / 0.31%**；是**开放环**评测，无 CoT / GRPO |
| nuScenes 附录 Table 13 | ST-P3 L2 **0.30 m** | 同表 OpenDriveVLA-3B ST-P3 碰撞率 **0.10%**，低于本文 0.12%；不能笼统断言所有指标绝对第一 |
| Prompter 预测提示 / oracle 真值提示（Table 9） | **90.4 / 90.8 PDMS** | +0.4 为不合法部署 oracle 对照，不是推理时默认可用的真值 |
| NAVSIM N=12 + DrivorR selector（Appendix 0.A） | **92.7 PDMS** | 非单次推理；**不得**直接替代官方单次 90.4 榜单数值 |
| H100 双模型 / 单模型双轮问答（Table 8，SFT-only） | **3.03 s / 2.87 s** | 对应 PDMS 88.2 / 87.0；双模型增加时延，单模型减性能 |

论文训练实现设置：8×H100、batch 16、AdamW `lr=4e-5`、weight decay 0.05，DVD 1 epoch、CoT-SFT 6 epoch、GRPO 180 steps（group 8）。这些是**论文协议**，不是已经在本地运行过的实验。

### 可靠性、源码与风险核对

- DVD 只在 20% 的 bbox 随机删除/抖动噪声条件仍保持正收益；40% 删除时 PDMS **85.3**，低于无 DVD 的 **86.2**（Table 7）。检测器失配会破坏指导。
- 数据构建的真实代码链路：`data_engine/main.py` 解析 YAML 并调用 `pipelines.runner.run_pipeline`，由 NAVSIM Adapter 抽取数据，`build_sample_ir` 构造 IR、`compute_sample_trajectory` 计算轨迹，再按注册的 enrichers 顺序加工，PromptBuilder 输出 JSON。
- `dvd/README.md` 区分 **DVD 自蒸馏 → Prompter SFT → Planner SFT** 三个本地训练入口，另有论文的 GRPO 后训练；官方训练 YAML 的 `dvd_h_size: 16` / `dvd_w_size: 28` 与论文所称 `2×4` 最优分块不一致，应复核默认配置及其实际计算语义后再复现。
- 主仓列出 `data_engine/`、`dvd/`、`sft/`，GRPO 说明与代码指向独立 Curious-VLA；README 未提供经验证的端到端车载部署基准。

**一手参考：** [论文 HTML v1](https://arxiv.org/html/2607.01658v1) · [官方仓库 README](https://github.com/ShivaTeam/DriveTeach-VLA/blob/master/README.md) · [数据引擎](https://github.com/ShivaTeam/DriveTeach-VLA/blob/master/data_engine/README.md) · [DVD 训练](https://github.com/ShivaTeam/DriveTeach-VLA/blob/master/dvd/README.md) · [数据引擎 Runner](https://github.com/ShivaTeam/DriveTeach-VLA/blob/master/data_engine/src/pipelines/runner.py) · [DVD 配置](https://github.com/ShivaTeam/DriveTeach-VLA/blob/master/dvd/distill_tgp_prompter.yaml)
