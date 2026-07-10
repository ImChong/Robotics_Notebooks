---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2411.02214"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_dexhub-and-dart.md
summary: "构建通才机器人系统受制于多样高质量数据的稀缺。本文提出 DART：一个借云端仿真与增强现实（AR）做可扩展机器人数据采集的众包遥操作平台。采集的数据自动存入 DexHub ——一个云端托管数据库，意在成为机器人学习的公共仓库。用户研究表明 DART 相比真机遥操作实现更高采集吞吐、更低体力疲劳，并能成功 sim-to-real 迁移、对视觉扰动鲁棒。"
---

# DexHub and DART

**DexHub and DART: Towards Internet Scale Robot Data Collection** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

构建通才机器人系统受制于多样高质量数据的稀缺。本文提出 DART：一个借云端仿真与增强现实（AR）做可扩展机器人数据采集的众包遥操作平台。采集的数据自动存入 DexHub ——一个云端托管数据库，意在成为机器人学习的公共仓库。用户研究表明 DART 相比真机遥操作实现更高采集吞吐、更低体力疲劳，并能成功 sim-to-real 迁移、对视觉扰动鲁棒。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| DART | 众包遥操作采集平台（云仿真 + AR） |
| DexHub | 云端公共机器人数据仓库 |
| Crowdsourcing | 众包 |
| Cloud Simulation | 云端仿真 |
| AR | 增强现实 |
| Throughput | 采集吞吐量 |

## 为什么重要

- **"云仿真 + AR 众包"是互联网规模采集的可行路径**，绕开本地硬件；
- **公共数据库**对社区共享与基础模型训练意义重大；
- 对人形（硬件稀缺）尤其友好；
- 与 ARMADA（AR 无机器人采集）思路相通、规模更大。

## 解决什么问题

通才机器人缺**互联网规模**数据： - 真机遥操作**吞吐低、疲劳高、难规模化**； - 缺**公共、可众包**的采集平台与仓库。

DexHub/DART 要：用**云仿真 + AR 众包**采集，建**公共数据库**，迈向互联网规模。

## 核心机制

1. **DART 众包采集平台**：云仿真 + AR，可扩展、低门槛；
2. **DexHub 公共数据库**：意在成为机器人学习公共仓库；
3. **优于真机遥操作**：吞吐更高、疲劳更低；
4. **sim-to-real + 视觉鲁棒**：采集数据可迁移真机。

方法拆解（深读笔记小节）：DART：云仿真 + AR 众包遥操作；DexHub：云端公共数据库；验证；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection.html> |
| arXiv | <https://arxiv.org/abs/2411.02214> |
| 作者 | Younghyo Park、Jagdeep Singh Bhatia、Lars Ankile、Pulkit Agrawal（MIT） |
| 发表 | 2024 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_dexhub-and-dart.md](../../sources/papers/humanoid_pnb_dexhub-and-dart.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection.html>
- 论文：<https://arxiv.org/abs/2411.02214>

## 推荐继续阅读

- [机器人论文阅读笔记：DexHub and DART](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection.html)
