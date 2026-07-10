---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2503.10554"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_nuexo.md
summary: "从动捕/遥操作到机器人技能学习的演进，是具身智能的关键路径。但现有系统难同时达成四目标：准确（长时间精确跟踪全上肢）、舒适（贴合人体生物力学）、通用（多模态采集如力数据、兼容人形）、便携（轻量户外日用）。NuExo 是一套可穿戴上肢外骨骼，配沉浸式直观遥操作与多模态感知采集来弥合此差距。凭借带同步连杆与同步带传动的新型肩部机构，它能很好地适配复合肩部运动，100% 覆盖自然上肢活动范围；整机仅 5.2 kg，支持背包式户外日常使用。作者还开发了统一直观的遥操作框架与多模态数据采集系统，兼容多款人形。跨平台、跨用户实验验证了其在运动范围与灵活性上的优势，以及在动态场景下数据采集与遥操作精度的稳定性。"
---

# NuExo

**NuExo: A Wearable Exoskeleton Covering all Upper Limb ROM for Outdoor Data Collection and Teleoperation of Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

从动捕/遥操作到机器人技能学习的演进，是具身智能的关键路径。但现有系统难同时达成四目标：准确（长时间精确跟踪全上肢）、舒适（贴合人体生物力学）、通用（多模态采集如力数据、兼容人形）、便携（轻量户外日用）。NuExo 是一套可穿戴上肢外骨骼，配沉浸式直观遥操作与多模态感知采集来弥合此差距。凭借带同步连杆与同步带传动的新型肩部机构，它能很好地适配复合肩部运动，100% 覆盖自然上肢活动范围；整机仅 5.2 kg，支持背包式户外日常使用。作者还开发了统一直观的遥操作框架与多模态数据采集系统，兼容多款人形。跨平台、跨用户实验验证了其在运动范围与灵活性上的优势，以及在动态场景下数据采集与遥操作精度的稳定性。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| ROM | Range of Motion，活动范围 |
| Exoskeleton | 外骨骼，可穿戴机械结构 |
| Synchronized Linkage | 同步连杆，适配复合肩部运动 |
| Timing Belt | 同步带传动 |
| Multi-modal Sensing | 多模态感知（含力数据） |
| Backpack-type | 背包式，便携户外 |

## 为什么重要

- **肩部复合运动是上肢外骨骼的难点**：新机构覆盖全 ROM 提升采集质量；
- **力等多模态数据**对接触/灵巧操作学习价值大；
- **便携户外**让数据采集走出实验室，扩大数据多样性；
- 与 ACE、CHILD 等可穿戴/外骨骼遥操作工作互补。

## 解决什么问题

上肢遥操作/采集设备难**同时**满足： - **准确**（长时间全上肢精确跟踪）； - **舒适**（贴合生物力学）； - **通用**（多模态采集 + 兼容人形）； - **便携**（轻量户外）。

尤其**肩部复合运动**难覆盖。NuExo 要：一套四目标兼顾的可穿戴上肢外骨骼。

## 核心机制

1. **全上肢 ROM 外骨骼**：新型肩部机构 100% 覆盖自然上肢活动范围；
2. **四目标兼顾**：准确、舒适、通用、便携（5.2 kg 背包式户外）；
3. **多模态采集（含力）+ 统一遥操作框架**：兼容多款人形；
4. **跨平台/跨用户验证**：运动范围、灵活性、动态稳定性。

方法拆解（深读笔记小节）：新型肩部机构（全 ROM）；轻量便携（5.2 kg 背包式）；统一遥操作框架 + 多模态采集；跨平台跨用户验证；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/NuExo__A_Wearable_Exoskeleton_Covering_all_Upper_Limb_ROM/NuExo__A_Wearable_Exoskeleton_Covering_all_Upper_Limb_ROM.html> |
| arXiv | <https://arxiv.org/abs/2503.10554> |
| 作者 | Rui Zhong、Chuang Cheng、Junpeng Xu、Yantong Wei、Ce Guo、Daoxun Zhang、Wei Dai、Huimin Lu（国防科大等） |
| 发表 | 2025 年 3 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_nuexo.md](../../sources/papers/humanoid_pnb_nuexo.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/NuExo__A_Wearable_Exoskeleton_Covering_all_Upper_Limb_ROM/NuExo__A_Wearable_Exoskeleton_Covering_all_Upper_Limb_ROM.html>
- 论文：<https://arxiv.org/abs/2503.10554>

## 推荐继续阅读

- [机器人论文阅读笔记：NuExo](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/NuExo__A_Wearable_Exoskeleton_Covering_all_Upper_Limb_ROM/NuExo__A_Wearable_Exoskeleton_Covering_all_Upper_Limb_ROM.html)
