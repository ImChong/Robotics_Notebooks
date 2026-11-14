# MOSS: Mode-Oriented Source Shaping for Robot Flow Matching Policies

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** MOSS: Mode-Oriented Source Shaping for Robot Flow Matching Policies
- **中文：** MOSS：从初始噪声选定 Flow Matching 动作模式
- **arXiv：** 2610.22193
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已阅读 arXiv HTML 的摘要、背景、方法、实验、结论、局限及可用附录；公开源码仅核查官方 README 与实际入口，未独立运行。
- **通读：** 已完成
- **论文：** https://arxiv.org/abs/2610.22193
- **HTML：** https://arxiv.org/html/2610.22193
- **PDF：** https://arxiv.org/pdf/2610.22193
- **官方项目页：** https://mmmmazr.github.io/moss/

## 论文机制

把训练中的动作轨迹标注为不同语义模式；依据模态强度估计器识别分叉区域，适配源高斯的间距：相似行为处合并，关键分叉处拉开；推理选对应分量初始噪声进入同一 flow network。

## 实验与证据边界

6 个仿真与 3 个真机任务，论文报告目标模式命中率最高可达 100%，10:90 模式不均衡时少数模式引导率超过 90%。模式命中率与任务完成成功率不是同一个指标。

## 复现与开放状态

项目页显示 Code 入口，但实际链接为 github.com/你的用户名/你的仓库名 占位，公开代码待核实。

## 局限

动作模式需要训练前定义，依赖模式标注和分叉检测；项目页 Code 链接指向示例占位地址，不能认定可运行源码已公开。

## 对 wiki 的映射

- [MOSS：从初始噪声选定 Flow Matching 动作模式](../../wiki/entities/paper-moss-mode-source-shaping.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.22193)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
