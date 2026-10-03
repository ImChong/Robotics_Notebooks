---
type: entity
tags:
  - paper
  - humanoid
  - loco-manipulation
  - whole-body-control
  - vision-language-action
  - egocentric-vision
  - human-demonstration
  - pretraining
  - unitree-g1
  - alibaba
  - shanghai-innovation-institute
  - buaa
  - hkust
  - arxiv
status: complete
updated: 2026-10-03
arxiv: "2610.00438"
related:
  - ../tasks/loco-manipulation.md
  - ../concepts/world-action-models.md
  - ./paper-wb-wam.md
  - ./paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
sources:
  - ../../sources/papers/lambda0_arxiv_2610_00438.md
summary: "λ₀（arXiv:2610.00438）以 500 小时 HumanVerse-500 第一视角全身人类移动操作数据，进行三阶段预训练/迁移；在 SIMPLE 与 4 项真机任务上评估，代码、模型和数据尚未发布。"
---

# λ₀ / HumanVerse-500

**λ₀**（*Towards a General Humanoid Loco-Manipulation Model via Egocentric Whole-Body Human Data Pretraining*，[arXiv:2610.00438](https://arxiv.org/abs/2610.00438)）是一种全身 humanoid VLA：先从第一视角人类数据学习互动，再用 **HumanVerse-500** 学习身体与手物互动的协调，最后适配人形机器人和下游任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 将视觉、语言指令和动作放在同一策略中建模 |
| RGB | Red-Green-Blue | 相机采集的彩色视频；论文使用第一视角视频作为人类数据的一部分 |
| G1 | Unitree G1 humanoid robot | 论文的下游人形机器人平台 |

## 为什么重要

- **把全身协调作为人类数据监督目标：** 从第一视角互动视频中恢复身体与手部运动，补足只监督手腕或手部动作时缺失的身体移动和姿态变化。
- **把「人类经验可扩展」变成可检验的训练配方：** 三阶段训练分别学习互动、全身协调和机器人适配，能通过数据规模与阶段消融分析每部分的作用。
- **评测连接仿真和真机：** 论文同时报告 SIMPLE 仿真结果与 4 项真机 loco-manipulation 任务，检验人类全身数据向机器人控制的迁移。

## 核心信息

| 项 | 内容 |
|---|---|
| **作者** | Chongyang Xu、Zhao Wu、Jin Chen、Yiming Jiang、Jinhui Ye、Yuming Jiang、Shifeng Zhang、Ziliang Feng、Mu Xu、Yilun Chen、Li Lu、Steven C. H. Hoi |
| **机构** | 阿里巴巴集团、四川大学、上海创智学院、北京航空航天大学、香港科技大学 |
| **预印本日期** | 2026-09-30（arXiv v1） |
| **人类数据** | HumanVerse-500：500 小时、32,993 episodes、827 类任务、约 8,700 万 pose 帧 |
| **机器人平台** | Unitree G1；论文称头部使用 GoPro HERO13 |
| **评测** | SIMPLE 仿真与 4 项真机 loco-manipulation 任务 |
| **公开状态（2026-10-03）** | arXiv 未列 GitHub、权重或数据链接；论文称后续发布 |

## 三阶段训练流程

```mermaid
flowchart LR
  ego["Stage I<br/>第一视角互动数据"] --> body["Stage II<br/>HumanVerse-500 全身协调"]
  body --> robot["Stage III<br/>机器人与任务适配"]
  robot --> g1["Unitree G1<br/>移动操作"]
```

1. **互动预训练：** 从不同来源的第一视角数据学习视觉互动模式。
2. **全身 mid-training：** 用 HumanVerse-500 学习身体运动、姿态和手物互动的时间协调。
3. **机器人 post-training：** 将共享的人类经验表示适配到目标机器人状态、动作接口和下游任务。

模型通过共享表示传递人类经验；不同 embodiment 的输入状态和动作空间差异由对应接口处理。

## 数据与迁移

HumanVerse-500 以轻量穿戴式系统同步采集第一视角视频和身体运动，再从视频恢复手部运动。论文将其作为 Stage II 的全身数据来源，作用是学习**身体移动与手物互动的配合关系**。这与只把人体手腕轨迹 retarget 到机械臂的监督方式不同，目标是支持需要接近、调整姿态、抓取和搬运的全身任务。

论文将数据集描述为开放世界、多任务的人类移动操作数据，并使用 Unitree G1 做下游人形验证。摘要与当前 arXiv 页面没有给出公开数据下载入口，因此这些数据规模是论文报告值，不等同于当前可下载的数据集。

## 实验与评测读法

- **仿真：** 在 SIMPLE 上评测，成功率比论文纳入比较的最强基线高 **3.9 个百分点**。
- **真机：** 在 4 项 loco-manipulation 任务上，成功率比论文纳入比较的最强基线高 **22.5 个百分点**。
- **数据扩展：** 作者报告 whole-body human data 增加时验证损失下降；验证的是论文所覆盖的数据规模范围，不应外推为任意数据量下都保持线性收益。
- **阶段消融：** whole-body mid-training 对真机成功率的贡献高于另一人类数据阶段；人类示范也提高了机器人未见物体上的任务进度。

## 源码运行时序图

**不适用（截至 2026-10-03）：** arXiv 未列可运行代码仓库、训练脚本或部署入口；论文称代码、模型和数据将来发布。目前无法据公开资料核对源码执行路径。

## 与其他工作对比

| 维度 | λ₀ | [WB-WAM](./paper-wb-wam.md) | [EgoMI](./paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md) |
|---|---|---|---|
| 人类数据角色 | 第一视角全身人类数据作为预训练 / mid-training 主体（HumanVerse-500） | 异构身–手数据预训练世界动作模型 | 第一视角人类演示直接用于模仿学习 |
| 模型形态 | 全身人形 VLA（三阶段训练） | 世界动作模型（WAM），联合预测未来与动作 | 主动视觉 + 双臂操作策略 |
| 机器人平台 | Unitree G1（腿足人形） | 人形 loco-manipulation | Rainbow RBY1（轮式半人形） |
| 开放状态 | 代码 / 权重 / 数据待发布 | 见实体页 | 代码 coming soon |

相对 EgoMI 只监督头部与双臂，λ₀ 把**身体移动与姿态**也纳入人类数据监督；相对 WB-WAM，λ₀ 不显式预测未来观测，而是把人类经验压进共享 VLA 表示再适配机器人。

## 结论

**这项工作支持一个明确判断：高覆盖的人类第一视角全身数据可以补足机器人遥操作数据稀缺，并改善人形移动操作迁移；当前证据仍以论文报告为准，尚待开放实现复现。**

- **主要增量是数据和训练阶段设计：** HumanVerse-500 提供同步的人类第一视角、身体与手部监督；Stage II 将全身协调作为专门训练阶段。
- **看结果时保留评测边界：** 3.9 个百分点来自 SIMPLE 仿真，22.5 个百分点来自 4 项真机任务；两者分别对照论文自己的最强基线。
- **不要把数据规模等同于公开可用：** 截至入库日期没有数据集下载链接，500 小时数据无法据当前页面独立获取。
- **复现关键待确认：** HumanVerse-500 的具体许可、标注格式、训练代码、模型权重以及 G1 适配接口尚未从公开链接核实。

## 局限与风险

- **源码与数据待发布：** 论文承诺后续发布，但目前 arXiv 页面没有提供链接；三阶段训练和评测尚不能按官方实现复现。
- **真机任务覆盖有限：** 论文摘要报告 4 项真实任务，不能据此推断对所有人形操作任务都泛化。
- **机构与规模信息按预印本记录：** 后续版本若更新数据划分、评测协议或开源状态，应以新版本为准。

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [World Action Models](../concepts/world-action-models.md)
- [WB-WAM：异构身–手预训练的人形 loco-manipulation WAM](./paper-wb-wam.md)
- [EgoMI：第一视角人类演示与全身操作](./paper-notebook-egomi-learning-active-vision-and-whole-body-mani.md)
- [本类论文索引：Loco-Manipulation and WBC](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源

- [来源归档：λ₀ / HumanVerse-500](../../sources/papers/lambda0_arxiv_2610_00438.md)
- [arXiv 摘要](https://arxiv.org/abs/2610.00438)
- [arXiv HTML 正文](https://arxiv.org/html/2610.00438)
- [arXiv PDF](https://arxiv.org/pdf/2610.00438)

## 推荐继续阅读

- [arXiv HTML 正文](https://arxiv.org/html/2610.00438)
- [WB-WAM](./paper-wb-wam.md) — 对照另一种使用人类数据进行人形全身预训练的路线
