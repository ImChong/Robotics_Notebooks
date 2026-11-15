# XBot-Omni：面向具身理解与交互的全模态模型（arXiv:2611.17160）

> 来源归档（ingest）｜XBot-Omni: An Omni-Modal Model for Embodied Understanding and Interaction

- **标题：** XBot-Omni: An Omni-Modal Model for Embodied Understanding and Interaction（XBot-Omni：面向具身理解与交互的全模态模型）
- **类型：** paper / omni-modal / embodied interaction / co-speech gesture / facial expression
- **作者：** Kerui Gu、Xing Fu、Shuoyang Sun、Cheng Peng、Jingru Duan、Huayu Gu、Zhenyu Chen、Jiaxin Huang、Cheng Wen、Rui Zhen、Kairui Chen、Zhihui Peng、Shaoli Huang（Project Lead）
- **机构：** 智元机器人 X-Lab（AGIBOT X-Lab）
- **版本：** arXiv:2611.17160v1，2026-11-12
- **论文：** <https://arxiv.org/abs/2611.17160>（[HTML 全文](https://arxiv.org/html/2611.17160) · [PDF](https://arxiv.org/pdf/2611.17160)）
- **项目页：** <https://xbot-omni.github.io/>
- **代码：** 2026-11-15 核查时未找到官方公开训练、推理或部署代码链接（项目页仅显示 Coming Soon）
- **数据：** 论文 §4.1 描述公开数据与私采数据混合；未查到该项目公开的数据下载入口
- **入库日期：** 2026-11-15
- **核对范围：** 直接通读 arXiv HTML v1 §1–7（架构 §2、原生状态与语音动作对齐 §3、数据与分阶段训练 §4、Table 1–8 的实验及评测限制 §5、相关工作 §6、结论 §7；核对图 1–2 文字说明）
- **通读：** 已完成
- **一句话说明：** Thinker–Talker–Actor 架构把文本/视觉/音视频理解、流式语音、语义手势、视觉目标指向和表情生成放入共享原生表征的具身交互系统。

## 核心论文摘录

### 1. 从 Thinker–Talker 扩展到 Actor（§2–3、Fig. 2）

- **Thinker：** 融合文本、图像、音频、视频与对话历史，给出回复并提供语义及场景隐状态；音视频 patch 有文本格式绝对时间戳。
- **Talker：** 借助自回归声学 token 及 codec 流式合成语音；向 Actor 暴露尚未波形解码的原生生成隐状态。
- **Actor 身体分支：** Semantic Action Controller 将 Thinker 语义读出为带类别、参数、文本锚点的事件；Speech–Action Aligner 根据 Talker 的原生状态估计语音进度；Motion Generator 依赖动作历史、事件条件与 Talker 语音状态逐步输出离散运动码，再解码为连续姿态。
- **Actor 表情分支：** Thinker/Talker 共享状态 → Chunk Transformer → 条件 Flow Matching → 连续表情序列，最后与语音/身体动作共用时间线。
- **对 wiki 的映射：** [XBot-Omni 论文实体](../../wiki/entities/paper-xbot-omni.md)；[统一多模态 token](../../wiki/methods/unified-multimodal-tokens.md)。

### 2. 视觉指向需要明确三维目标和词语时序（§3.2–3.3、§4.4）

- Thinker 判断用户所指的目标；视觉定位与深度估计经相机内参和 camera-to-body 外参变换，生成**身体坐标系的三维目标**。
- 目标位置作为语义事件的空间条件，相关指称词语成为时序锚点；Speech–Action Aligner 估计说到该短语的时刻，让动作提前准备并同步表达。
- Q1 / X2 的平台专属运动学适配器把运动参考与目标映射到关节轨迹，使用指向轴标定和逆运动学；**不是**直接输出通用机器人关节控制指令。
- **对 wiki 的映射：** [XBot-Omni 论文实体](../../wiki/entities/paper-xbot-omni.md)。

### 3. 分阶段训练与数据边界（§4）

- 数据：公开具身活动数据 + 私有的人—人、人—机器人交互及同步语音—动作录制；agent 用 ASR、说话人分离、检测/跟踪、动作识别、音视频定位做自动标注与跨模态核验。
- Thinker：多模态 **SFT** → 按领域用 **GRPO** 优化多个专家 → **MOPD** 将专家能力蒸馏入同一个学生 Thinker。专家仅用于训练，部署时使用单个整合后的 Thinker。
- Actor：原生 Talker 隐状态适配/语音对齐 → 语义事件及结束条件学习 → 模型生成动作历史和组合事件训练；表情分支采用条件 Flow Matching。
- 指向训练（SGPT）：目标定位 → 带三维条件的事件学习；此阶段 Thinker 和 Talker 冻结。
- **对 wiki 的映射：** [XBot-Omni 论文实体](../../wiki/entities/paper-xbot-omni.md)。

### 4. 论文实验快照（§5、Table 2–8）

| 实验 / 指标 | XBot-Omni | 参照与读数条件 |
|-------------|-----------|----------------|
| Daily-Omni 音视频准确率 | **86.30%** | Qwen3.5-Omni-Plus 84.74%；差 **+1.56 pp** |
| ReTurn 五轮交互 MCQ | **84.00%** | 最强对照 70.04%；Reconfirm / Retrieve 并非最优 |
| ERIBench 空间关系 | **59.17%** | Qwen3.5-Omni-Plus 38.18%；意图推断 85.51% 低于基线 91.45% |
| BEAT2 参考动作分布 FGD（越低越好） | **1.351** | EMAGE 1.389；以统一身体表示评测 |
| 语义手势 SQ（0–3） | **1.383** | 最强对照 0.608；以模型辅助的匿名视频评分为准 |
| 条件语义时序 CT | **85.19%（46/54）** | 仅 119 个有效短回复中可判定清楚手势且有有效对齐的 **54** 个，不能写成全部回应成功率 |
| SGPT 指向联合时空 STG | **0.843** | 无 SGPT 0.241；100 条经选择的开发集短片 |
| 首次动作延迟中位数 | **527 ms** | 集成系统报告值；不代表完整任务完成时间或全硬件最坏时延 |

**对 wiki 的映射：** [XBot-Omni 论文实体](../../wiki/entities/paper-xbot-omni.md)；[Daily-Omni 音视频基准](../../wiki/entities/paper-daily-omni.md)。

## 原文声明与不能外推的边界

- 论文在 **PrimeBOT Q1** 与 **AGIBOT X2** 展示了语音、动作、表情及指向交互，但未提供可据以替代商用真机可靠性评估的跨任务闭环成功率。
- 指向任务的 STG 来自有筛选条件的 **100 条开发集样本**；动作 FGD 依赖统一编码器及特定人类动作参考分布。
- SQ 是机器辅助评分；CT 的分母随模型可判定手势而变，不能直接作为严格同样本成功率对照。
- 目前无法审查代码实现、训练数据细目、运行配置或实际机器人安全约束；主页公开内容截至 2026-11-15 为 Coming Soon。

## 对 wiki 的映射

- 主实体：[XBot-Omni](../../wiki/entities/paper-xbot-omni.md)
- 项目页归档：[xbot-omni-github-io.md](../sites/xbot-omni-github-io.md)
- 对照：[Daily-Omni](../../wiki/entities/paper-daily-omni.md)、[统一多模态 token](../../wiki/methods/unified-multimodal-tokens.md)
