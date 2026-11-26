---
type: method
tags: [cad, engineering, geometry]
status: complete
updated: 2026-11-26
summary: "通过几何约束、参数和可编辑操作历史构造并检查工程模型。"
sources:
  - ../../sources/repos/multi-agent-cad.md
  - ../../sources/sites/platform-method-parametric-cad-workflow.md
original:
  - ../../sources/sites/platform-method-parametric-cad-workflow.md
related:
  - engineering-simulation-workflow.md
  - ../concepts/text-to-cad.md
---

# Parametric CAD Workflow · 参数化 CAD 建模流程

## 一句话定义

通过几何约束、参数和可编辑操作历史构造并检查工程模型。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| CAD | Computer-Aided Design | 计算机辅助设计 |
| B-Rep | Boundary Representation | 以面、边等边界表示实体 |
| STEP | Standard for the Exchange of Product Model Data | 工程产品数据交换标准 |
| STL | Stereolithography | 常见三角网格交换格式 |

## 为什么重要

能打开 STEP 或 STL 只说明几何可交换，不等于恢复尺寸约束、参数和操作历史。

## 主要技术路线（核心机制）

CadQuery 分别提供 Workplane、Sketch 和 Assembly 接口，底层使用 OCCT 的 B-Rep。以参数化脚本控制尺寸，在工作平面构造或修改特征，随后定位零件组成装配。STEP 导出供几何交换；CadQuery 完整参数化信息留在 Python 源档。

```mermaid
flowchart LR
    N0["需求与尺寸"]
    N1["定义基准和参数"]
    N2["草图与特征"]
    N3["再生和装配检查"]
    N4["导出与下游复核"]
    N0 --> N1 --> N2 --> N3 --> N4
```

## 工程实践

为孔径、孔距、厚度等使用命名参数，测试名义值和边界值；检查孔是否贯穿、最小壁厚与装配干涉。版本化保留脚本和参数表。导出 STEP 后，在接收端量测一个已知尺寸；下游若需要 STL，另记录网格化公差。

## 局限与风险

拓扑选择可能因上游特征重排而指向另一处；格式和内核版本也会影响重建。参数化不能替代公差分析、工艺审查或结构验证。

## 关联页面

- [工程仿真建模与验证](engineering-simulation-workflow.md)
- [文本到 CAD](../concepts/text-to-cad.md)

## 参考来源

- [CadQuery Concepts 与 Importing and Exporting Files](../../sources/sites/platform-method-parametric-cad-workflow.md)

## 推荐继续阅读

- [几何及接口](https://cadquery.readthedocs.io/en/latest/primer.html)
- [导入导出](https://cadquery.readthedocs.io/en/latest/importexport.html)
