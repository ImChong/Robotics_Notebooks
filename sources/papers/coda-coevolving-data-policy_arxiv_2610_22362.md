# CoDA: Co-Evolution of Data Engines and Action Policies

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** CoDA: Co-Evolution of Data Engines and Action Policies
- **中文：** CoDA：失败驱动的数据引擎与动作策略协同进化
- **arXiv：** 2610.22362
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已直接阅读 arXiv HTML 摘要、方法与部分实验/局限章节；尚未逐行检查全文及全部附录，后续继续核对
- **通读：** 未完成
- **论文：** https://arxiv.org/abs/2610.22362
- **HTML：** https://arxiv.org/html/2610.22362
- **PDF：** https://arxiv.org/pdf/2610.22362
- **官方项目页：** https://plan-lab.github.io/projects/coda/

## 论文机制

记录 rollout 的阶段化失败与场景有效性，诊断为 difficulty mode；同一诊断编译成新的场景生成程序和 Text2potential 奖励/价值学习信号，并按学习增益调度数据生成预算。

## 实验与证据边界

四类操纵任务 OOD 成功率 27.6%，对应 SFT 基线 5.3%、按任务成功率调度的数据引擎 18.1%；LIBERO-Pro 的位置扰动结果 25%→84%、指令扰动 1%→76%。主要为仿真结果，不能外推为真机效果。

## 复现与开放状态

官方项目页 2026-11-14 检查：Code (coming soon)，尚未提供可用源码地址。

## 局限

依赖可用仿真环境、可验证失败诊断及场景生成程序；项目页标注 Code coming soon，完整复现暂受限。

## 对 wiki 的映射

- [CoDA：失败驱动的数据引擎与动作策略协同进化](../../wiki/entities/paper-coda-coevolving-data-policy.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.22362)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
