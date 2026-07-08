# GaP（Graph-as-Policy）项目页归档

> 来源：<https://graph-robots.github.io/gap/>（ingest 快照）

- **标题：** GaP — Graph-as-Policy
- **类型：** project-site
- **论文：** arXiv:2607.05369
- **入库日期：** 2026-07-08

## 站点要点（与论文一致）

- **定位：** 面向 **变体自动化（VA）** 的 **多 agent 自学习 harness**；自然语言任务 → **有向计算图**（感知/抓取/搬运/恢复路由），仿真与真机 **同一图** 执行。
- **任务谱：** **FA**（固定重复）→ **VA**（有界几何/位姿变化，GaP 目标）→ **GR**（开放通才 VLA）。
- **交互 demo：** 杂货打包 `workflow.json` 可点击节点查看代码、输入与路由；含 success/failure route 与回环直至全部打包完成。
- **自学习：** Make Popcorn 初图 **33%** → 10 轮仿真排练 **94%（sim）/ 90%（real 18/20）**；三阶段为抓取改进、搬运调整、放置精修。
- **8 VA benchmark：** VA-I 杂货履约、VA-II 杂货打包、VA-III 做爆米花、VA-IV USB-C 插线（UR5+力反馈）、VA-V 工业洗箱（双臂）；前六项 Franka + LIBERO 衍生场景。
- **MORSL：** 51 项初始技能；混合 ROS 过程与 GraspGen 等 model-free 原语。
- **结果亮点：** 大位姿变化列 GaP **0.93–0.99**；VLA 降至 **~0.20**；**π₀.₅ w/ GaP** staging **>2×** 裸 VLA；无图单 agent CaP **0%**。
- **代码/数据：** 论文注明将发布于项目页（ingest 时以站点与 arXiv 为准）。

## 对 wiki 的映射

- [GaP 论文实体](../../wiki/entities/paper-gap-graph-as-policy.md)
- [变体自动化概念](../../wiki/concepts/variational-automation.md)
- [sources/papers/gap_arxiv_2607_05369.md](../papers/gap_arxiv_2607_05369.md)
