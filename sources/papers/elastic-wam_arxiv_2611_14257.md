# Elastic-WAM: Adaptive Spatiotemporal Imagination for Efficient Robot Control / 自适应时空想象的高效机器人控制

> 原始论文来源归档；这里只归纳论文的公开方法、数据与评测边界，不替代全文。

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14257
- **全文：** https://arxiv.org/html/2611.14257v1
- **版本：** arXiv:2611.14257v1（2026-11）
- **核对范围：** HTML 全文：Introduction、Related Work、Methodology 3.1–3.4、Experiments 4.1–4.4、Conclusion；Appendix A–C（实现、额外仿真、真机评测）。
- **通读：** 已完成
- **收录日期：** 2026-11-12
- **项目页：** https://lihaitian.com/Elastic-WAM/

## 论文概要

Elastic-WAM 是为世界–动作模型动态分配视频未来想象计算的控制方法，决定何时停止、做多少步、更新哪些区域。

## 机制要点

- 学习时空精炼需求场，动态决定未来视频区域刷新与停止时机。
- RoboCasa-GR1 官方报告 57.4% / 388.2ms，相比 Joint-WAM 55.0% / 572.6ms；其他对照并不全部更慢。
- 无已核实的官方实现入口，尚不能重跑实际时延/性能实验。

## 评测边界

论文列出 LIBERO 98.8%、LIBERO-Plus 73.7%、LIBERO-Pro 54.1%；RoboCasa-GR1 成功率 57.4%，相应推理约 388.2ms，所比较 Joint-WAM 为 55.0% / 572.6ms。RoboCasa 上并非比所有优化基线都快，Fast-WAM 在论文对比中约 331ms。真实机器人茶饮等任务展示了不同干扰条件下的表现，仍需结合任务表看泛化。

## 通读原文：方法、实验与附录复核

### 自适应场与三阶段学习

将 **196** 个未来视频 token 分为 **64** 个组；共享 Beta 形时间基函数，叠加区域权重。全局 demand 的累计 commit probability 跨阈值 **η=0.5** 触发停止；时间基的 mode 决定下一去噪跨度；**K=96** 个 future token 的空间刷新预算由 region score 排序选择。冻结原 WAM 骨干，按 HOW FAR→WHERE→WHEN 三阶段以终端 action error 监督场参数。

### 推理与实验对照

每轮先一次 dense joint denoising，后续按需求稀疏刷新 video token，其余 KV 和隐状态复用；world commit 后继续动作去噪。表 IV：RoboCasa-GR1 **57.4% / 388.2ms**，Joint-WAM **55.0% / 572.6ms**，Fast-WAM **54.9% / 331.1ms**；测速 GPU **H800**，不是比所有基线最快。LIBERO **98.8%**、LIBERO-Plus **73.7%**、LIBERO-Pro **54.1%**。

### 真机与局限

附录 C：双臂 UFACTORY xArm 7 四阶段泡茶，**200** 条示范，名义/光照 OOD/对象 OOD 各 **20** 次 rollout。名义成功率 **50%（Joint-WAM 30%）**，对象 OOD 则与 Joint-WAM 均为 **30%**。部分说明是作者假设，不是已识别的因果机制；仍无已核实源码入口。

## 仍待独立验证

- 论文成功率与延迟属于作者报告结果，尚未在本地复现实验；需检查代码和数据版本是否可获取。

## 对 wiki 的映射

- [Elastic-WAM](../../wiki/entities/paper-elastic-wam.md)
