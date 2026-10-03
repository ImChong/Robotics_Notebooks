# λ₀ / HumanVerse-500（arXiv:2610.00438）

> 来源归档（ingest）

- **标题：** Towards a General Humanoid Loco-Manipulation Model via Egocentric Whole-Body Human Data Pretraining
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2610.00438>
- **HTML：** <https://arxiv.org/html/2610.00438>
- **PDF：** <https://arxiv.org/pdf/2610.00438>
- **TeX 源码：** <https://arxiv.org/src/2610.00438>
- **作者：** Chongyang Xu、Zhao Wu、Jin Chen、Yiming Jiang、Jinhui Ye、Yuming Jiang、Shifeng Zhang、Ziliang Feng、Mu Xu、Yilun Chen、Li Lu、Steven C. H. Hoi
- **机构：** Alibaba Group；Sichuan University；Shanghai Innovation Institute；Beihang University；The Hong Kong University of Science and Technology
- **项目页 / 代码 / 权重 / 数据：** 截至 **2026-10-03**，arXiv 页面未列独立项目页、代码仓库、权重或数据链接；论文称后续将发布代码、模型和数据。
- **提交日期：** 2026-09-30（arXiv v1；预印本）
- **一句话说明：** 提出 HumanVerse-500（500 小时第一视角、身体与手部同步的人类移动操作数据）和全身 humanoid VLA λ₀；通过三阶段训练，把人类互动经验迁移到 Unitree G1 的 loco-manipulation。

## 核心摘录（MVP）

### 1) 问题与贡献

- **摘录要点：** 第一视角人类视频规模大、包含丰富物体互动，但常见手腕/手部监督无法覆盖人形机器人接近物体、调整身体姿态、抓取并搬运时所需的全身协调。机器人全身遥操作演示则采集成本高。
- **论文贡献：** 提出 **HumanVerse-500**，一个记录第一视角视频、身体运动和手部运动的 500 小时人类 loco-manipulation 数据集；提出 **λ₀**，通过共享表示和 embodiment-specific 接口把人类经验迁移到人形 VLA。
- **对 wiki 的映射：**
  - [λ₀ / HumanVerse-500](../../wiki/entities/paper-lambda0-egocentric-human-pretraining.md) — 论文实体页
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 全身移动操作任务

### 2) HumanVerse-500 数据

- **摘录要点：** 论文报告数据集含约 **500 小时、32,993 段 episode、827 类任务、约 8,700 万 pose 帧**，覆盖开放世界中的多样化人类移动操作；穿戴式采集系统同步记录第一视角视频与身体运动，再从视频恢复手部运动。
- **平台信息：** 下游实体为 **Unitree G1**，头部安装 **GoPro HERO13**。
- **对 wiki 的映射：**
  - [EgoMI](../../wiki/entities/paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md) — 第一视角人类演示与机器人操作

### 3) λ₀ 三阶段训练

- **Stage I — 第一视角互动预训练：** 从多样化 egocentric 数据学习手部与物体互动。
- **Stage II — 全身人类数据 mid-training：** 使用 HumanVerse-500 学习身体移动、姿态变化与手物互动之间的协调。
- **Stage III — 机器人 post-training：** 适配下游任务和机器人本体；人和机器人状态/动作差异由 embodiment-specific 接口处理，模型共享表示空间。
- **对 wiki 的映射：**
  - [Whole-Body WAM](../../wiki/entities/paper-wb-wam.md) — 全身动作与人形 loco-manipulation 相关工作
  - [World Action Models](../../wiki/concepts/world-action-models.md) — 视觉与动作联合建模语境

### 4) 评测结果与解释

- **摘录要点：** 在 SIMPLE 仿真评测及 4 项真机 loco-manipulation 任务中评估；论文报告相对最强已评估基线，成功率分别提高 **3.9 个百分点**与 **22.5 个百分点**。
- **作者的分析：** 随着 whole-body human data 增加，验证损失下降；消融实验显示 whole-body mid-training 对真机成功率贡献更大；人类示范也改善了对机器人未见物体的任务进度。
- **对 wiki 的映射：**
  - [Humanoid Loco-Manipulation & WBC 分类页](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

### 5) 开源状态（截至 2026-10-03）

- **核查结论：** arXiv 页面没有列项目主页、GitHub、Hugging Face、模型权重或数据集链接；论文只说明将发布代码、模型和数据，当前按 **待发布** 记录。
- **可用资料：** arXiv abstract、HTML、PDF 和 TeX 源码均可公开访问。
- **对 wiki 的映射：**
  - [λ₀ / HumanVerse-500](../../wiki/entities/paper-lambda0-egocentric-human-pretraining.md) — 开源边界与复现状态

## 当前提炼状态

- [x] 核实 arXiv 摘要、HTML 与版本日期
- [x] 记录 HumanVerse-500、三阶段训练和论文报告的评测增益
- [x] 核查公开的项目页、源码、模型与数据入口；当前未列出
- [x] wiki 映射：`wiki/entities/paper-lambda0-egocentric-human-pretraining.md` 新建
