# RoboTalk: Learning Multi-Robot Communication and Coordination from Multimodal Demonstrations

- **作者：** Dorian Benhamou Goldfajn*, Mason Nakamura*, Saaduddin Mahmud, Justin Svegliato, Kyle H. Wray, Shlomo Zilberstein（* 同等贡献）
- **机构：** University of Massachusetts Amherst（麻省大学阿默斯特分校）；University of California, Berkeley（加州大学伯克利分校）；Northeastern University（美国东北大学）
- **版本 / 载体：** arXiv:2609.23997v2 [cs.RO]（Submitted on 21 Sep 2026 v1，last revised 8 Oct 2026 v2，arXiv HTML 全文）
- **原文：** https://arxiv.org/abs/2609.23997 （全文读取载体：https://arxiv.org/html/2609.23997 ）
- **项目页 / 代码：** 已开源 — 官方代码与复现仓库 [DorianAtSchool/robotalk](https://github.com/DorianAtSchool/robotalk)（RoboTalk 代码、数据集与模型权重采用 Apache-2.0 协议；内含修改版 `robocasa/` 采用 MIT 协议，另配套修改版依赖 [DorianAtSchool/robosuite](https://github.com/DorianAtSchool/robosuite) 与 Hugging Face 数据集 [DorianAtSchool/RoboTalk](https://huggingface.co/datasets/DorianAtSchool/RoboTalk)）
- **核对范围：** 论文全文全部章节（Abstract、Sec. I–VI、Table I–IV、Fig. 1–7 图注及正文），以及官方 GitHub 仓库 `README.md`、模块结构与实验数据表
- **通读：** 已完成

## 原文核对笔记

- **核心问题与建模：** 研究部分可观测（Dec-POMDP）条件下的双移动操作机器人去中心化协作，聚焦 8B 参数量级可端侧部署的小型视觉–语言模型（VLM），使其从多模态示范中联合学习自然语言通信与高层技能工具调用（感知、导航、操作、通信）。每个智能体仅维护私有历史（自身相机观测、工具调用记录、收到的消息），并在同一步（timestep）以该步开始时的符号状态并行校验动作。
- **双机器人仿真与三大协调机制：** 基于 RoboCasa365 改编出 53 个多阶段厨房移动操作任务（43 个训练任务 + 10 个留出泛化任务；无单阶段任务，Phase 2/3/4 分布为训练 12/12/19、留出 2/2/6），在 4 种布局 × 5 种视觉风格 × 3 个随机种子的兼容场景中运行。引入有限状态机（FSM）校验动作前置条件与协作约束，并设计三项任务无关协调机制：
  1. **Leader–Follower 规划协议（Table II）：** 物理执行前先完成两步握手通信（Leader 提议分工 → Follower 确认），再并行执行各自子任务。
  2. **Wait–Release 同步机制：** 存在跨机器人前置依赖时，等待方先发消息声明条件与精确释放关键词，再调用 `wait_for_signal` 挂起，直到搭档发送含匹配关键词的 `communicate` 消息后恢复。
  3. **独占工作区访问规则：** 冰箱、灶台/台面电器等窄空间一次仅允许一台机器人进入，占用方须调用 `give_space` 或移离交互区域（公共台面与餐桌可共享）。
- **三阶段数据合成与采样多样性（Table III）：**
  - 先由 Gemini-3-Flash 生成双机联合轨迹（工具调用、自然语言消息、一句话推理轨迹 rationale），经 FSM 校验并以报错反馈重试；再于仿真中重放插入 `get_image` 观测调用并渲染相机视角；最后拆分为单智能体上下文训练样本，共建成 **7,950 条轨迹**（每任务 150 条；HF `ticks` 表含 66,592 个时间步）。
  - 采样对比（30 条/任务）：**Structured Random** 采样在通信嵌入多样性（`Qwen3-Embedding-4B` 余弦距离）上相对 Base 提升 **+52.00%**（0.088427 vs. 0.058174），显著高于 Verbalized（+33.98%）、High Temp（+8.05%）和 Random UUID（+2.15%），故被选作主数据集采样方案；Verbalized 采样在动作序列归一化 Levenshtein 多样性上提升最高（+28.25%，0.458226 vs. 0.357283）。
- **闭环评测指标与核心实验结论（Fig. 5–7）：**
  - 指标为闭环**无错 FSM 成功率**（在参考解长度 4 倍预算内达成目标且全程零工具调用失败/被拒；固定评测集每任务 10 局，共 430 局训练任务 + 100 局留出任务）。
  - **RQ1 通信引导消融（Fig. 5）：** 未微调 `Gemini-3-Flash` 在 None / Unguided / Minimal / Intermediate / Full 五档下的训练/留出成功率为 7.7%/5%、9.5%/8%、43.3%/34%、46.0%/46%、65.6%/72%；而未微调 `Qwen3-VL-8B-Instruct` 在全部五档下仅 0.0%–1.6%（训练）与 0%（留出），证明小模型仅靠通信通道或提示协议无法可靠协调。
  - **RQ2 数据规模与 SFT（Fig. 6）：** LoRA 微调（rank=16, alpha=32, lr=1e-4, 1 epoch, batch=64）后，`Qwen3-VL-8B-Instruct` 在训练任务随每任务轨迹数（30→60→90→120→150）单调提升至 **81.9% → 87.9% → 92.1% → 93.0% → 94.2%**，留出任务为 **45% → 64% → 55% → 48% → 65%**（非单调）。`Qwen3-VL-8B-Thinking` + rationale 训练在训练任务达 **80.2% / 84.7% / 91.4% / 90.2% / 92.6%**，留出任务全面领先达 **71% / 72% / 66% / 77% / 73%**，在 120 条/任务时取得最高 **77%** 留出成功率。
  - **RQ3 推理轨迹消融（Fig. 7，30 条/任务）：** 引入 rationale 监督使 Instruct 留出成功率从 **45%** 升至 **58%**（训练 81.9%→82.6%），使 Thinking 从 **50%** 升至 **71%**（训练 71.6%→80.2%）。
- **局限与边界：** 低层执行采用脚本化仿真技能而非连续真机电机策略；成功判定依赖抽象物理细节的符号 FSM（例如把锅铲放进锅中即记为搅拌）；增加训练任务轨迹量并不保证留出新任务泛化单调上升；目前限于仿真双机厨房场景。
- **对 wiki 的映射：** [RoboTalk：多模态示范驱动的双机器人通信与协作学习](../../wiki/entities/paper-agentic-2609-23997.md)
