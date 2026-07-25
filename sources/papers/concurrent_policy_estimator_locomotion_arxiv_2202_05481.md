# Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion

> 来源归档

- **标题：** Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion
- **类型：** paper
- **作者：** Gwanghyeon Ji, Juhyeok Mun, Hyeongjun Kim, Jemin Hwangbo
- **机构：** KAIST
- **链接：** https://arxiv.org/abs/2202.05481
- **PDF：** https://arxiv.org/pdf/2202.05481
- **入库日期：** 2026-07-25
- **一句话说明：** 策略与状态估计网络**并发训练**：策略输出期望关节位置，估计器输出基座线速度/足高/接触概率等；快速仿真训练后迁移真机，穿越山坡、滑板、可变形地面等。
- **开源状态：** 论文引用/讨论 Cheetah-Software 生态；**截至入库日无与本文一一对应的官方独立训练仓钉死**（社区有相关 fork）。按「部分依赖开源仿真/控制栈、训练代码未官方单列」处理。
- **沉淀到 wiki：** [paper-concurrent-policy-estimator-locomotion](../../wiki/entities/paper-concurrent-policy-estimator-locomotion.md)
- **注：** 与 `sources/papers/privileged_training.md` 条目 3 同主题；正确 arXiv 为 **2202.05481**（非 2202.05738）。

---

## 核心贡献（摘录）

1. 单阶段并发：策略与估计器互相提供训练信号，弱化两阶段 teacher–student 串行。
2. 估计量直接服务动态鲁棒 loco。
3. 多样地形真机迁移。

## 对 wiki 的映射

- [paper-concurrent-policy-estimator-locomotion](../../wiki/entities/paper-concurrent-policy-estimator-locomotion.md)
- [privileged-training](../../wiki/concepts/privileged-training.md)
- [state-estimation](../../wiki/concepts/state-estimation.md)
