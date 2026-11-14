---
type: entity
project_id: urdf2ontology
arxiv: "2611.13850"
code: https://github.com/RIS-WITH/urdf2ontology
tags: [paper, urdf2ontology, robotics]
status: draft
updated: 2026-11-14
summary: "URDF2Ontology 将 URDF 的运动学图通过双轨形态解析、LLM 语义推理和图清理转成 OWL 本体。"
related:
  - ../concepts/motion-retargeting.md
  - ../concepts/whole-body-control.md
original:
  - ../../sources/papers/urdf2ontology_arxiv_2611_13850.md
sources:
  - ../../sources/papers/urdf2ontology_arxiv_2611_13850.md
  - ../../sources/repos/urdf2ontology.md
---

# Taxonomy-Agnostic Neurosymbolic Grounding of Robot Kinematics｜URDF2Ontology 机器人运动学语义接地

## 一句话定义

URDF2Ontology 将 URDF 的运动学图通过双轨形态解析、LLM 语义推理和图清理转成 OWL 本体。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| URDF | Unified Robot Description Format | 本研究涉及的URDF术语 |
| OWL | Web Ontology Language | 本研究涉及的OWL术语 |
| LLM | Large Language Model | 本研究涉及的LLM术语 |

## 为什么重要

读取关节/连杆图和用户提供的 taxonomy。

## 核心原理与方法

1. 读取关节/连杆图和用户提供的 taxonomy。
2. 分支结构用 bottom-up Tree Track，串联结构用 top-down Chain Track。
3. 对已解析子图逐步清理，通过符号规则约束 LLM 输出语义能力本体。

## 流程总览

```mermaid
flowchart LR
  A0["URDF/目标本体"]
  A1["Tree/Chain Track"]
  A2["LLM 语义赋值"]
  A3["图逐步清除"]
  A4["OWL 输出"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

> 方法流程根据摘要/官网整理，尚待论文原文逐图核实。

## 实验与评测

摘要称异构机器人上能抽取结构、实现 schema 对齐；文章提及五类机器人，复杂传感器和语义分类仍有局限。

上述结果均为作者报告或用户附件转载，**不是本次复测**。

## 结论

**URDF2Ontology 将 URDF 的运动学图通过双轨形态解析、LLM 语义推理和图清理转成 OWL 本体。**

1. 读取关节/连杆图和用户提供的 taxonomy。
2. 分支结构用 bottom-up Tree Track，串联结构用 top-down Chain Track。
3. URDF 缺少任务语义，LLM 语义推断可能幻觉；不等同自动认证机器人能力。

## 与其他工作对比

比较本文与相关基线时应在相同任务、数据、样本量、传感器和部署条件下评估；不能把论文中的单个数字解释为真实部署的通用进步。待全文表格核对后补足逐项量化对照。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant A0 as main.py
  participant A1 as URDF graph
  participant A2 as Tree/Chain Track
  participant A3 as OWL/ontologenius
  A0->>A1: 启动评测/训练
  A1->>A2: 处理环境与模型输入
  A2->>A3: 执行模型或输出指标
  A3-->>A0: 返回状态
````

**README 入口：** `python3 main.py（前置 ontologenius 与模型配置）`。图仅表示已核对到的入口与概念性组件顺序，不代表已运行或逐行审阅。

## 工程实践

- 对齐版本、传感器、动作/观测空间和评测预算后再尝试复现。
- 正式应用应增加跨分布扰动与任务安全边界检查；原文完整复现步骤待核对。
- 本次通读状态：**未完成**；页面仍为 draft。

## 局限与风险

URDF 缺少任务语义，LLM 语义推断可能幻觉；不等同自动认证机器人能力。

未对全部论文正文/附录及运行细节完成审阅，任何进一步量化外推均不成立。

## 关联页面

- [motion-retargeting](../concepts/motion-retargeting.md)
- [whole-body-control](../concepts/whole-body-control.md)

## 参考来源

- [原论文及阅读进度](../../sources/papers/urdf2ontology_arxiv_2611_13850.md)
- [官方源码入口](../../sources/repos/urdf2ontology.md)
- [用户附件 13 篇原始资讯整理](../../sources/blogs/embodied-research-digest-2026-11-12.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.13850)
- [官方源码](https://github.com/RIS-WITH/urdf2ontology)
