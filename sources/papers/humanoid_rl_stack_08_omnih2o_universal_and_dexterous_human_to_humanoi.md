# OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning

> 来源归档（ingest · 人形 RL 身体系统栈 42 篇 · 第 08/42）

- **标题：** OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning
- **类型：** paper
- **系统栈层：** 01 数据 · 重定向 · 遥操作
- **机构：** CMU；上海交通大学
- **会议：** CoRL 2024
- **arXiv：** <https://arxiv.org/abs/2406.08858>
- **代码：** <https://github.com/LeCAR-Lab/human2humanoid>
- **项目/论文链接：** <https://omni.human2humanoid.com/>
- **索引来源：** [具身智能研究室 · 42 篇 RL 运动控制长文](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）
- **原始抓取：** [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **核对日期：** 2026-11-22
- **核对范围：** `arXiv:2406.08858` 正文 §1–§5（含 Table 1–3、OmniH2O-6 数据集与 Limitations）及项目主页 `https://omni.human2humanoid.com/`；官方仓库与 H2O 共仓 `LeCAR-Lab/human2humanoid`（`legged_gym/legged_gym/scripts/train_hydra.py`）
- **通读：** 已完成
- **一句话说明：** 以运动学姿态为统一全身控制接口，将特权全状态 Teacher（PPO）通过 DAgger 蒸馏至仅有头部与双手 3 点稀疏目标及 25 步本体历史的 Student，支持 VR/RGB/语言/GPT-4o/模仿学习与双手灵巧操作。

## 核心摘录（一手原文 + 策展）

- **在身体系统栈中的位置：** 01 数据 · 重定向 · 遥操作，编号 **08/42**。
- **Teacher–Student 管线：** Teacher 观测 $S \subset \mathbb{R}^{913}$ 含特权状态，PPO 训练于过滤后的 AMASS 子集 $\hat{Q}$；Deployable Student $\pi_{\text{OmniH2O}}$（$S \subset \mathbb{R}^{1665}$）移除全局线速度（真机 VIO 会导致跌倒，Table 2a），输入 25 步本体历史与头部+双手 3 点全局位置 $\tilde{p}_t$，经 DAgger 蒸馏（仿真成功率 94.10% vs H2O 87.52%、去 DAgger 仅 47.11%；真机 $E_{\text{g-mpjpe}}$ 47.94 mm vs H2O 87.33 mm）。
- **OmniH2O-6 与自主控制：** 采集 6 个日常双足+灵巧手任务共 40 min 30 Hz RGBD+运动目标+关节目标数据，Diffusion Policy（DP-DDPM）在 4 个任务平均成功率 8/10（序列观测+序列动作 `Se-O-Se-A` 达 8.75/10），显著高于 ResNet BC（1/10）；另可接入 GPT-4o 原语选择与 MDM 文本生成运动控制。
- **总框架见：** [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)。

## 对 wiki 的映射

  - [OmniH2O 论文实体](../../wiki/entities/paper-hrl-stack-08-omnih2o.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 姊妹篇 AMP 专题：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
