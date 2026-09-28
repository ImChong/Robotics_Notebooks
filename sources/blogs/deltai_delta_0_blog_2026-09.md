# Delta-0 (Δ₀): A New Chapter in Humanoid Intelligence

> 来源归档（blog / 公司官方）

- **标题：** Delta-0 (Δ₀): A New Chapter in Humanoid Intelligence
- **类型：** blog
- **作者 / 组织：** Delta Intelligence（德塔智能）
- **原始链接：** <https://deltai.com/en/blog/delta-0>
- **发表日期：** 2026-09（页眉标注 September 2026）
- **入库日期：** 2026-09-28
- **一句话说明：** 官方发布人形基础模型 Δ₀：脑（潜空间 world–action MoT）与 69-DoF 全身控制器协同设计，强调人数据预训练、real-to-sim-to-real 评测与 delta-action 真机 RL。

## 开源状态（步骤 2.5，2026-09-28）

- **确认未开源**：核对 [deltai.com 首页](https://deltai.com/en) 与本文页眉/页脚，**无 GitHub、Hugging Face 或论文 PDF 链接**。控制器对比实验引用 **Motion Tracking Leaderboard 配套代码** 跑 HEFT / MimicLite / SONIC / ScaleBFM XL 等基线，**不等于** Δ₀ 权重或训练栈公开。

## 核心摘录（归纳，非全文）

### 问题定位

- 将 **general、可靠的 whole-body loco-manipulation** 视为人形智能核心未解问题：浮动基座、手–眼–足耦合、长时程误差累积。
- 六条能力轴：**全身灵巧（69 DoF）**、**单通才策略近人类速度**、**脑–控制器协同扩展**、**大规模人数据预训练**、**real-to-sim-to-real**、**delta-action 人机闭环 RL**。

### 系统结构

- **Brain**：潜空间 **world–action model**；**MoT** 三分支——vision-language（多视角 + 语言）、vision（DINO 语义空间预测未来视觉特征）、action（本体条件下全身 motion）。
- **四种训练模式**（切换条件与去噪目标）：forward dynamics、inverse dynamics、visual planning、policy-only。
- **Controller**：学习式 **whole-body controller**，把 brain 的 **motion command** 转为 **69-DoF 关节目标**；**delta-action** 接口同时服务策略输出与人机纠正；支持遥操作与自主执行共享表示。
- **长时程**：靠上层模型/智能体/人类提供 **stage-level 指令**；Δ₀ 负责阶段间 **重定位、站姿、抓放保持** 等物理过渡（非独立高层任务规划器）。

### 数据与动作空间

- Controller 在 **大规模高质量人 motion** 上训练；brain 预训练 **>10,000 h** 配对 **egocentric 观测 + 全身 motion**。
- **共享动作表示**：核心 **154 维**（臂/手、root command、motion command 等），**padding 至 180 维**；多数据源/多 embodiment（单臂、双臂、ego、UMI、wholebody motion、人形机 motion）共用通道布局。

### 评测与缩放（文内自报）

- **控制器 zero-shot 跟踪**（日常任务 motion 集，约 **3 h 31 min**）：相对 Motion Tracking Leaderboard 协议，报告 **Global Root Error** 与 **MPJPE**；称优于 HEFT、MimicLite v1.1、SONIC v1.1、ScaleBFM XL。
- **Coverage**：日常任务 motion 中 wrist ≤2 cm 且 root ≤15 cm 的占比随 motion 数据预算上升。
- **策略预训练**：0% / 10% / 100% 人数据三档，任务成功率随预训练增加（分基础 loco-manip、双手精细、全身接触三类）。
- **Real-to-sim-to-real**：视频重建仿真资产（文称结合 **GPT-6 Astra** 等 LLM agent 管线）；洗碗机任务 sim 与真机成功率随训练数据 **同向变化**。
- **真机 RL**：value 模型估计 progress；HIL correction chunk 与自主 rollout 共用 delta-action 接口。洗碗机 **镜像厨房 OOD**：SFT **4/20** → HIL + RL 后 **13/20**。

### 演示任务（视频叙事）

- 铺床、地面拾物、开洗碗机、脚踏垃圾桶、坐沙发、黑胶转盘操作等；另含光照变化、漏抓恢复、人为干扰下自主完成等片段。

- **对 wiki 的映射：** [delta-0-humanoid-foundation-model](../../wiki/entities/delta-0-humanoid-foundation-model.md)
