# On Agentic Manipulation Policies — Apple AMP1（arXiv:2611.09892）

> 来源归档（ingest；以作者论文为一手资料，不以社交媒体配图为事实依据）

- **标题：** On Agentic Manipulation Policies
- **类型：** paper
- **机构：** 苹果公司（Apple）
- **作者：** Chen Chen, Omid Bagherieh, Naveen Nikamanth Appiah Balaji, Jesse T. Gonzalez, Negin Heravi, Fahad Islam, Xueting Li, Jean-Baptiste Regli, Kejia Ren, Petr Švec, David J. Yoon, Matthias Müller, Vladlen Koltun
- **论文：** <https://arxiv.org/abs/2611.09892>；[HTML 原文](https://arxiv.org/html/2611.09892)；[PDF](https://arxiv.org/pdf/2611.09892)
- **项目页：** <https://apple-aiml-research.github.io/ml-amp/>
- **官方代码：** <https://github.com/apple-aiml-research/ml-amp>
- **版本：** arXiv:2611.09892（2026-11 的公开版本；以页面版本记录为准）
- **入库 / 核验日期：** 2026-11-10
- **核对范围：** 通读论文 HTML 的引言、方法、RoboLab-120/CaP-Gym/真机/RoboDojo 实验、消融、讨论与附录；交叉核对官方项目页同仓 `docs/index.html`、README、复现文档及关键运行入口；未在本地重跑实验。
- **通读：** 已完成
- **一句话说明：** AMP1 将通用前沿模型接入机器人感知、几何和运动 SDK，通过持续编写与执行 Python 代码，在观察—计划—行动—核验循环中完成操作；AMP1+ 则以练习形成持久任务记忆。

## 论文核心内容与证据边界

### 1. 操作架构：模型写代码，而非直接输出关节动作

- **原文位置：** 方法概览、系统接口、附录实现细节。
- **摘录要点：** 通用模型在持久 Python 环境里调用 harness SDK：读取相机 RGB-D、标定和机器人状态，操作可校正的场景表示，发出定时末端位姿轨迹与夹爪命令，再根据实际反馈修正。底层 SDK/机器人服务负责机器人接口与运动执行。与直接输出低层电机命令或离线训练单一 VLA 策略不同。
- **对 wiki 的映射：** [AMP1 论文实体](../../wiki/entities/paper-apple-amp1.md)、[智能体操作策略方法](../../wiki/methods/agentic-manipulation-policies.md)。

### 2. 零样本与多基准实验

- **原文位置：** RoboLab-120、CaP-Gym 和实物机器人实验表/图。
- **摘录要点：**
  - RoboLab-120：AMP1 + Gemini 4 Argon **85.6%**、AMP1 + Claude Opus 5.5 **82.8%**（按论文评测设置；120 任务、每任务 10 次）。
  - LIBERO-PRO（CaP-Gym）**95.2%**；robosuite **100%**；BEHAVIOR-1K **96.0%**。这几项有独立的任务/设置，不可视作统一机器人能力评分。
  - 真机 12 任务 × 9 次 = 108 次：AMP1 + Opus 5.5 **104/108（96.3%）**；真机平均完成耗时约 280 秒，其中推理约 211 秒、物理执行约 69 秒。
- **对 wiki 的映射：** [AMP1 论文实体](../../wiki/entities/paper-apple-amp1.md) 的评测与局限。

### 3. 训练外记忆：AMP1+ 的自我改进

- **原文位置：** RoboDojo 实验、practice 与 memory 分析。
- **摘录要点：** AMP1+ 不是新训练的权重；在练习中以文字工作笔记和代码片段积累任务经验，评测中复用。RoboDojo 结果从无记忆零样本 **26.4%** 提升至 few-shot practice 后 **84.1%**，持续练习设定报道 **57.6%**。不要混淆零样本 26.4% 与练习后 84.1% 的条件。
- **对 wiki 的映射：** [AMP1 论文实体](../../wiki/entities/paper-apple-amp1.md) 与 [智能体操作策略方法](../../wiki/methods/agentic-manipulation-policies.md)。

### 4. 基座模型差异与 harness 差异不可混为一谈

- **原文位置：** RoboLab-120 比较、agent/harness 对比分析。
- **摘录要点：** 文中提到 Astranomy / GPT-6 Astra **74.8%**，而 AMP1 + Gemini 4 Argon **85.6%**；但两者使用不同模型与不同 harness，不能据此推断单独的模型排名或系统是否“没上线”。作者的同底座模型子集比较也显示 harness 的设计对结果影响巨大。
- **对 wiki 的映射：** [智能体操作策略方法](../../wiki/methods/agentic-manipulation-policies.md) 的选型判断。

### 5. 消融、局限及可复现性

- **原文位置：** sensing/vision 消融、讨论与局限；对应官方 `docs/reproducibility.md`。
- **摘录要点：** 对 Opus 5.5，完整传感方案约 **82.8%**、RGB-only 约 **75.8%**、文本化视觉约 **36.8%**；更多高级感知模块未必总能提升结果。模型推理耗时仍大、接触实时性受限，研究主要针对相对静态的操作任务。官方代码依赖外部仿真基准及固定版本，硬件/驱动会影响复现。
- **对 wiki 的映射：** [AMP1 论文实体](../../wiki/entities/paper-apple-amp1.md)、[操作任务](../../wiki/tasks/manipulation.md)。

## 开源边界（核查于 2026-11-10）

- **已开源：** 官方 `apple-aiml-research/ml-amp` 发布 `amp` CLI、RoboLab/CaP-Gym/RoboDojo runner、agent loop、harness SDK 与详细复现说明。
- **第三方依赖：** RoboLab、CaP-X、RoboDojo 需要另行克隆，按官方 `docs/reproducibility.md` 固定 commit；API 模型与部分场景资产须按各方许可证单独取得。
- **未独立验证：** 未运行 H100 实验，不应把作者结果表述为本仓重测结果。
- **资料互指：** [项目页归档](../sites/apple-aiml-research-ml-amp.md) · [官方仓归档](../repos/apple-ml-amp.md)。

## 对 wiki 的映射（汇总）

- [Apple AMP1 / AMP1+ 主实体](../../wiki/entities/paper-apple-amp1.md) — 论文与源码统一整理。
- [Agentic Manipulation Policies 概念与工程方法](../../wiki/methods/agentic-manipulation-policies.md) — 对比 VLA / 固定技能编排路线。
