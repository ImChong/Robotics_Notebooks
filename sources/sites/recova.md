# Recova 官方项目主页（agent-guided failure recovery）

> 来源归档（site）

- **标题：** Recova: Agent-Guided Failure Recovery for Autonomous Robotic Manipulation
- **类型：** site
- **项目主页：** <https://www.liuisabella.com/Recova/>
- **NVIDIA 官方镜像：** <https://research.nvidia.com/labs/gear/recova/>
- **论文：** <https://arxiv.org/abs/2610.01178>
- **入库日期：** 2026-11-07
- **核对范围：** 官方页面 Overview / DAgger data collection / Digital twins / Recovery skills / Simulation benchmarks / Fail-and-recover / Citation；核验页面实际公开资源。
- **通读：** 已完成
- **一句话说明：** 官方页面以演示视频、数字孪生回放、失败恢复对照和基准结果说明 Recova 的 real-to-sim-to-real 恢复闭环。

## 可用的一手资源

- 论文链接、四工位真机数据采集演示、数字孪生可视化、仿真失败与恢复的成对片段。
- 部分仿真恢复实例可通过页面的 “View recovery script” 查看代码片段。
- 页面给出 LIBERO-Pro、MolmoSpaces 比较与 BibTeX；网页展示并非独立的训练/部署源码仓库。

## 开源状态核查（2026-11-07）

**官方完整实现尚未核实开放**：页面头部、内容与页尾未列 GitHub/模型权重/数据集下载入口；NVIDIA 同步项目页也未列。论文附录 A 有真机编排伪代码，但伪代码不是可运行发布包。若未来出现官方仓库，应补记 URL、许可证、commit 与实际运行入口，届时再绘制源码时序图。

## 与论文的关系

- [论文全文归档](../papers/recova_arxiv_2610_01178.md)
- [Recova 统一 wiki 节点](../../wiki/entities/paper-recova.md)
