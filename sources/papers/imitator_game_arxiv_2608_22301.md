# The Imitator Game（意图级模仿基准）

> 来源归档（ingest）

- **标题：** The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.22301>
- **机构：** 香港大学（HKU）；超忆（TranscEngram）；复旦大学（Fudan）；浙江大学（ZJU）
- **作者：** Xunzhe Zhou、Yiyang Cai、Fengyi Wang、Ran Ju、Hanxiang Ren、Ruizhe Liu、Yu Zhang、Qian Luo、Feng Chen、Pei Zhou、Yi Ma、Yanchao Yang
- **项目页：** <https://imitator-game.github.io/>
- **入库日期：** 2026-08-30
- **一句话说明：** L0–L3 四级基准把人类示范与机器人现场的差距逐步拉大，用目标等价而非动作相似衡量模仿；配套 IG-10K 与盲测 Arena。

## 核心摘录（MVP）

### 1) 轨迹复现 ≠ 意图模仿

- **摘录要点：** 现有策略学 observation→action，面对人类视频多在近似场景复现轨迹。真正的模仿是推断目标，并用手头工具/物体/布局完成同一意图。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)
  - [模仿学习](../../wiki/methods/imitation-learning.md)

### 2) 四级差距 + IG-10K

- **摘录要点：** L0 场景相同；L1 空间适应；L2 视觉泛化；L3 功能替代（不同 affordance 实现同一意图）。IG-10K：2 万余组人机配对、50+ 任务、6 领域，仿真与真机统一格式。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)

### 3) 评测数字

- **摘录要点：** 9 个先进模型 L0–L2 稳定，L3 崩溃。人视频条件优于字幕条件。未见任务零样本均 **<13%**；IG-10K 预训练后再用 10 组配对微调收益随预训练规模增大。自动成功率与 Arena 人类判断 r≈0.86。
- **对 wiki 的映射：**
  - [Imitator Game](../../wiki/entities/paper-imitator-game.md)

### 4) 开源状态（截至 2026-08-30）

- **摘录要点：** **部分开源**。项目页提供 Arena / 任务画廊 / 提交入口；未见独立可运行训练仓。数据集入口在项目页，复现前需再核下载是否开放。

## 当前提炼状态

- [x] 项目页与 arXiv 摘要对齐
- [x] wiki 映射：`wiki/entities/paper-imitator-game.md` 新建
