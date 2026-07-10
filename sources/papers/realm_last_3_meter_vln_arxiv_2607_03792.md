# From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation（arXiv:2607.03792）

> 来源归档（ingest）

- **标题：** From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation
- **类型：** paper / VLN / REVERIE / instance grounding / evaluation
- **arXiv：** <https://arxiv.org/abs/2607.03792>
- **PDF：** <https://arxiv.org/pdf/2607.03792>
- **作者：** Xiangyu Shi, Ruoxi Yang, Wei Tao, Jiwen Zhang, Yanyuan Qiao（Project Lead）, Qi Wu（通讯作者）
- **机构：** 阿德莱德大学澳大利亚机器学习研究所（AIML）、如一动力（Ruyi Dynamics）、复旦大学、洛桑联邦理工学院（EPFL）
- **入库日期：** 2026-07-10
- **一句话说明：** 指出连续 VLN（尤其 REVERIE-CE）在 **3 m 区域到达** 与 **实例级可见接地** 之间的 **Last-3-Meter Grounding Gap**；提出可插拔精修模块 **REALM**、实例中心指标 **ONS/GS/OracleGS** 与数据集 **REVERIE-AIM**（含 180K 短视界末段样本），在 ETPNav / UniNaVid / SmartWay 等四类骨干上一致提升细粒度接地，并在 Hello Robot Stretch 真机验证。

## 摘要级要点

- **问题：** 现有 REVERIE / REVERIE-CE 成功判据多继承 VLN-CE 的 **3 m 半径到达**，不评估 **最终朝向** 与 **目标可见性**；智能体可在墙侧或背对目标处停车仍算成功。
- **Last-3-Meter Grounding Gap：** 粗粒度 **region arrival** 与细粒度 **object instance grounding** 之间的系统性断裂；对后续检视、操作等下游任务，区域到达不是功能终点。
- **诊断指标（解耦导航与接地）：**
  - **ONS@τ** — 停车点到目标实例周围可导航点的最小测地距离（τ=0.1 m / 0.5 m）
  - **GS** — 最终视角开放词汇检测框与 GT 的 IoU≥0.5（与导航是否成功无关）
  - **OracleGS** — GT 框在最终相机帧中可见且面积占比≥1%
- **基线落差示例：** ETPNav-FT 在 REVERIE-AIM 上 SR=34.67%，但 ONS@0.1m 仅 6.32%、OracleGS 11.31%——超八成「成功」未能真正接地目标。
- **REALM（Region-to-Entity Alignment for Last-3-Meter Navigation）：** 上游任意 VLN 策略 **stop** 后接管；仅用 **egocentric RGB + 原指令**，无需深度/地图/改上游架构；三阶段：**导航 π_nav → 精修 π_ref → 接地（BERT 短语抽取 + OWLv2 框）**。
- **π_ref 实现：** 以 **UniNaVid** 为骨干，**LoRA** 微调 LLM 线性层；**Visibility-Aware Stop Penalty（VSP）** 惩罚目标不可见时的过早 stop。
- **REVERIE-AIM：** 将 REVERIE 离散图轨迹映射到 Habitat 连续空间；用 ObjectNav 协议把区域级终点换成 **实例几何边界/可视区域内的采样点**（平均最小测地距 1.32 m→0.34 m）；训练集 3,691 条长视界轨迹、验证 1,344 条；另生成约 **180K** 末段短视界 clip（锚点扰动 + 随机朝向）。
- **实验：** 四类上游（ETPNav-ZS/FT、UniNaVid-ZS、SmartWay）× 六种末段策略；REALM Full 在各骨干上全面提升 ONS/GS/OracleGS（如 ETPNav-ZS ONS@0.1m 7.07%→14.66%，相对 +107%）。真机 Stretch：ONS@0.5m 8.33%→33.33%。

## 核心摘录（面向 wiki 编译）

### REALM 三阶段流水线

1. **Navigation：** π_nav 跟随语言指令，在 s_T_nav 发出 stop。
2. **Refinement：** π_ref 用短视界动作 `{forward, turn_left, turn_right, stop}` 重定位到 s_T_ref，使目标 **既近又可见**。
3. **Grounding：** BERT-Large 抽取式 QA 从指令中提取目标名词短语 → **OWLv2** 开放词汇检测输出边界框。

### 与相邻工作的分界

| 路线 | 与 REALM 的分界 |
|------|-----------------|
| **GroundingMate** | 离散图 REVERIE；从预标注候选框中 **分类选择**，非开放检测；未针对连续末段评测协议 |
| **SLING / MSGNav / AnyImageNav / APRR** | 假设已定位目标或给定 goal image；未系统修正 VLN **仅看 3 m 距离** 的评测盲区 |
| **ETPNav / Dynam3D / D3D-VLP** | 长视界 VLN-CE 专用方法；REALM 作为 **plug-and-play** 末段模块叠加其上 |

### 关键实验数字（REVERIE-AIM val-unseen，索引级）

| 骨干 | 策略 | ONS@0.1m | ONS@0.5m | GS | OracleGS |
|------|------|----------|----------|-----|----------|
| ETPNav-ZS | 无精修 | 7.07 | 11.98 | 3.05 | 6.99 |
| ETPNav-ZS | REALM Full | **14.66** | **21.13** | **6.10** | **16.52** |
| ETPNav-FT | 无精修 | 6.32 | 12.20 | 4.46 | 11.31 |
| ETPNav-FT | REALM Full | **17.19** | **25.89** | **8.48** | **20.24** |
| Human | — | 50.40 | 74.71 | 38.24 | 80.38 |

## 对 wiki 的映射

- 沉淀实体页：[REALM · Last-3-Meter VLN 实例接地](../../wiki/entities/paper-realm-last-3-meter-vln-grounding.md)
- 交叉补强：[视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)、[REVERIE](../../wiki/entities/paper-vln-03-reverie.md)、[ETPNav](../../wiki/entities/paper-vln-09-etpnav.md)、[VLN 四范式复现路径](../../wiki/overview/vln-open-source-repro-paradigms.md)（UniNaVid 骨干）

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2607.03792>
