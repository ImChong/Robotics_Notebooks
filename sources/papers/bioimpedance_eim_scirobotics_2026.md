# Bioimpedance meets biomechanics: Wearable EIM encodes fascicle and activation dynamics（Science Robotics, 2026）

> 来源归档（ingest）

- **标题：** Bioimpedance meets biomechanics: Wearable electrical impedance myography encodes fascicle and activation dynamics
- **类型：** paper / wearable sensing / bioimpedance / electrical impedance myography (EIM) / assistive robotics / biomechanics
- **期刊：** Science Robotics, 2026（Vol. 11, Issue 118）
- **发表：** 2026-09-23（print）
- **DOI：** <https://doi.org/10.1126/scirobotics.aea4580>
- **数据与复现代码：** <https://doi.org/10.5281/zenodo.22044875>（CC BY 4.0；见 [`bioimpedance_eim_zenodo_22044875.md`](../repos/bioimpedance_eim_zenodo_22044875.md)）
- **项目页 / GitHub：** **无独立项目页或 GitHub 持续维护仓**（截至 **2026-09-25**；复现入口为 Zenodo 数据集 + Jupyter 笔记本）
- **Science.org 全文：** 入库环境访问 DOI 落地页返回 **403**；正文以 Crossref JATS 摘要 + Zenodo README 核对
- **作者：** Christopher J. Nichols、Nicholas Harris、H. Trask Crane、Autumn Routt、Milki D. Haile、Quentin Goossens、Minoru Shinohara、Gregory S. Sawicki、Kristen L. Jakubowski、Omer T. Inan
- **机构：** 佐治亚理工学院（Georgia Institute of Technology，School of Electrical and Computer Engineering 等；含 Inan 可穿戴传感方向）
- **代码与数据：** **部分开源** — 已发布 **处理后数据**（`Manuscript_Data.xlsx`）与 **图表复现笔记本**（`Manuscript_Code.ipynb`）；**无**原始采集固件、可穿戴硬件设计或实时控制栈
- **入库日期：** 2026-09-25
- **一句话说明：** 在功能性动态运动中，用生物力学对照（测力计、B 超肌束长度、sEMG 激活）建立 **双频 EIM** 与 **肌束长度动力学 / 激活** 的可解释关系；关节力矩相关仅见于部分工况；步行中 **双频 EIM + PCA** 可稳定跟踪肌束长度与激活变化，支撑辅助机器人控制与损伤预防传感。

## 相关资料

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.aea4580](https://doi.org/10.1126/scirobotics.aea4580) | Science Robotics 原文 |
| Crossref | [works API](https://api.crossref.org/works/10.1126/scirobotics.aea4580) | 元数据与 JATS 摘要 |
| Zenodo | [10.5281/zenodo.22044875](https://doi.org/10.5281/zenodo.22044875) | 补充数据 + 图表复现代码 |
| 代码归档 | [`bioimpedance_eim_zenodo_22044875.md`](../repos/bioimpedance_eim_zenodo_22044875.md) | 仓库内复现入口说明 |
| 多模态人体数据对照 | [HUMAPS-4D 实体](../../wiki/entities/paper-humaps4d.md) | sEMG + 足底 + MoCap；与 EIM 可穿戴叙事互补 |
| 遥操作任务 | [`teleoperation.md`](../../wiki/tasks/teleoperation.md) | 辅助 / 可穿戴传感进闭环控制的场景 |

## 摘要级要点

- **动机：** 辅助机器人控制需要 **可穿戴、非侵入** 传感；EIM 动态测组织阻抗，但既往多靠黑箱或 **孤立动作** 外推至肌力/疲劳，生理可解释性不足。
- **方法：** 内侧腓肠肌 **双频 EIM**；**10 名**受试者；同步 **测力计（关节力矩）**、**B 模式超声（肌束架构）**、**肌电（激活）**。
- **协议：** 约束等长、自选等长、向心、离心等多类动态收缩；多力级、多关节角、多角速度。
- **主要发现 1：** EIM 与 **关节力矩** 仅在 **特定条件** 下相关；与 **肌长度变化动力学** 在 **全部运动类型** 下相关。
- **主要发现 2：** 混合效应模型 + 顺序特征分析 → **肌束长度** 与 **肌肉激活** 为 EIM 方差 **主驱动**。
- **主要发现 3：** 步行中 **双频 EIM + 主成分分析（PCA）** 可 **可靠捕获** 肌束长度与激活变化。
- **应用指向：** 功能性运动中的 **实时生物力学传感**；辅助机器人控制、损伤预防。

## 核心摘录（面向 wiki 编译）

### 1) EIM 从「黑箱力估计」到「可解释肌束 + 激活」

- 摘要强调用 **生物力学原理** 与传统金标准对齐，而非单一孤立动作外推。
- **对 wiki 的映射：** 实体页「核心原理」— 区分 **力矩代理（条件性）** vs **长度动力学 + 激活（普适）**。

### 2) 实验对照栈

| 模态 | 角色 |
|------|------|
| 双频 EIM | 可穿戴阻抗；待解释信号 |
| 测力计 | 关节力矩（kinetics） |
| B 超 | 肌束长度 / 架构（kinematics） |
| sEMG | 神经肌肉激活 |

- **对 wiki 的映射：** Mermaid 流程图「采集 → 对齐 → 建模 → 步行 PCA」。

### 3) 开源与复现边界

- Zenodo：**处理后 Excel + 图表复现 notebook**；README 列 Python 3.11.10 与科学栈依赖。
- **非发布：** 原始波形级全量、硬件固件、在线 assistive 控制 demo。
- **对 wiki 的映射：** 「部分开源」；源码运行时序图仅覆盖 **Zenodo 图表复现路径**。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-bioimpedance-eim-wearable-myography.md`](../../wiki/entities/paper-bioimpedance-eim-wearable-myography.md)**
- 代码归档：**[`sources/repos/bioimpedance_eim_zenodo_22044875.md`](../repos/bioimpedance_eim_zenodo_22044875.md)**
- 交叉：**[`teleoperation.md`](../../wiki/tasks/teleoperation.md)**、**[`paper-humaps4d.md`](../../wiki/entities/paper-humaps4d.md)**（sEMG 多模态人体）
