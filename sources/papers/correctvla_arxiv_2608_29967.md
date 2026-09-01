# CorrectVLA（VLA 推理期语言动作纠错）

> 来源归档（ingest）

- **标题：** Training-Free Action Correction for VLA Model Failures via Language Feedback
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.29967>
- **作者：** Owen Kwon, Pablo Ortega-Kral, Arthur Bucker, Jean Oh
- **项目页：** <https://correctvla.github.io/>
- **代码：** <https://github.com/owenk3/correct_vla>
- **入库日期：** 2026-09-01
- **一句话说明：** 将任务级自然语言纠正转化为加性动作幅度调整，不修改策略权重；人类一次反馈统一应用于所有 rollout；适用于 execution misalignment，语义理解崩溃不适用。

## 核心摘录（MVP）

### 1) 失败模式边界

- **摘录要点：** VLA 部署仍有系统性失败；LIBERO-90 失败分类显示「目标正确但动作幅度失准」的 execution misalignment 为可修正子集，语义崩溃子集不适用。
- **对 wiki 的映射：**
  - [CorrectVLA](../../wiki/entities/paper-correctvla.md)

### 2) Training-free 动作修正

- **摘录要点：** CorrectVLA 把语言反馈映射为加性动作幅度调整；仿真分布内/OOD 可恢复错位；真机 xArm7 在基础策略近乎失效时恢复近完美成功率。
- **对 wiki 的映射：**
  - [CorrectVLA](../../wiki/entities/paper-correctvla.md)

### 3) 开源状态（截至 2026-09-01）

- **摘录要点：** **已开源** `owenk3/correct_vla`（项目页 Code 按钮指向该仓）。
- **对 wiki 的映射：**
  - [CorrectVLA 项目页](../sites/correctvla.md)
  - [owenk3/correct_vla](../repos/owenk3-correct-vla.md)

## 当前提炼状态

- [x] 项目页/仓库核查
- [x] wiki 映射：`wiki/entities/paper-correctvla.md` 新建
