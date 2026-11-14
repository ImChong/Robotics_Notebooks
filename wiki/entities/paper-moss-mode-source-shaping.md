---
type: entity
project_id: moss-mode-source-shaping
tags: [paper, robotics, ingest-2026-11]
status: draft
updated: 2026-11-14
arxiv: "2610.22193"
project: https://mmmmazr.github.io/moss/
related:
  - ./paper-hdh-action-verifier.md
  - ./paper-trace-conversational-humanoid-motion.md
original:
  - ../../sources/papers/moss-mode-source-shaping_arxiv_2610_22193.md
sources:
  - ../../sources/papers/moss-mode-source-shaping_arxiv_2610_22193.md
  - ../../sources/sites/moss-mode-source-shaping.md
summary: "行为模式可控的 Flow Matching 初始源分布"
---

# MOSS：从初始噪声选定 Flow Matching 动作模式

## 一句话定义

**将多模态机器人行为对应到可区分的高斯起点分布，在推理时通过选择初始源分量控制目标动作模式。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MOSS | Mode-Oriented Source Shaping | 面向动作模式的源分布塑形 |
| FM | Flow Matching | 连续生成流匹配 |
| ODE | Ordinary Differential Equation | 连续流运输方程 |

## 为什么重要

行为模式可控的 Flow Matching 初始源分布，关注现有控制、数据或机器人学习方法在复杂任务中的失效点。这里单独建论文实体，避免与其他论文共用详情页；当前是待完成全文与源码复核的草稿。

## 核心机制

把训练中的动作轨迹标注为不同语义模式；依据模态强度估计器识别分叉区域，适配源高斯的间距：相似行为处合并，关键分叉处拉开；推理选对应分量初始噪声进入同一 flow network。

## 方法流程（待全文逐节复核）

以下流程仅依据已核对的论文 HTML 方法描述，不代替源代码调用图。

```mermaid
flowchart LR
  demo[带语义模式的多模态示范] --> strength[模态强度估计]
  strength --> sources[自适应高斯源分布]
  demo --> flow[共享 Flow Matching 网络]
  sources --> flow
  choice[指定目标动作模式] --> sources
  flow --> action[按指定模式采样动作]
```

## 源码运行时序图

**待核实，不绘制未经代码阅读确认的时序图。** 目前未确认可运行官方源码，待项目源文件发布或访问恢复后补充。

## 实验与评测

6 个仿真与 3 个真机任务，论文报告目标模式命中率最高可达 100%，10:90 模式不均衡时少数模式引导率超过 90%。模式命中率与任务完成成功率不是同一个指标。

## 局限与工程实践

- 动作模式需要训练前定义，依赖模式标注和分叉检测；项目页 Code 链接指向示例占位地址，不能认定可运行源码已公开。
- **源码状态：** 项目页显示 Code 入口，但实际链接为 github.com/你的用户名/你的仓库名 占位，公开代码待核实。
- **原文状态：** 本轮已核查 arXiv 部分章节，尚未通读全部正文/附录，保持 draft，不宣称健康度满分。

## 结论

**若任务需在两种同样可行的动作路线中主动选择，控制源噪声分布比增加随机采样更直接。**

1. 使用与论文一致的任务、硬件和评测协议复验，不跨实验口径比较数据。
2. 把方法改善的指标与额外的数据、计算或控制成本分开记录。
3. 对未经独立核查的复现入口，暂不声称完成验证。

## 关联页面

- [HDH：冻结 VLA 的低延迟动作验证](./paper-hdh-action-verifier.md)
- [TRACE：具有参考保持约束的对话式人形动作编辑](./paper-trace-conversational-humanoid-motion.md)

## 参考来源

- [论文归档](../../sources/papers/moss-mode-source-shaping_arxiv_2610_22193.md)
- [官方项目页归档](../../sources/sites/moss-mode-source-shaping.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2610.22193)
- [官方项目页](https://mmmmazr.github.io/moss/)
