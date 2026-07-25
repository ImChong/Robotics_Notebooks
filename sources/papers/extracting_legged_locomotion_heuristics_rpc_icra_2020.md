# Extracting Legged Locomotion Heuristics with Regularized Predictive Control

> 来源归档

- **标题：** Extracting Legged Locomotion Heuristics with Regularized Predictive Control
- **类型：** paper
- **作者：** Gerardo Bledt, Sangbae Kim
- **机构：** MIT
- **链接：** https://ieeexplore.ieee.org/document/9197488
- **DOI：** https://doi.org/10.1109/ICRA40945.2020.9197488
- **会议：** ICRA 2020
- **入库日期：** 2026-07-25
- **一句话说明：** 离线充分探索代价空间，拟合「命令–最优动作–状态」简单模型以提取正则启发式，再在线自适应；不改控制器结构/增益即可在 Mini Cheetah 上增强能力。
- **开源状态：** **未单独开源**该方法工具链；RPC/控制实现可参考 Cheetah-Software 生态。
- **沉淀到 wiki：** [paper-extracting-legged-locomotion-heuristics-rpc](../../wiki/entities/paper-extracting-legged-locomotion-heuristics-rpc.md)

---

## 核心贡献（摘录）

1. 用仿真离线探索识别启发式候选。
2. 简单模型 + 在线参数适应，保留物理直觉。
3. Mini Cheetah 真机验证能力提升且无需改控制结构。

## 对 wiki 的映射

- [paper-extracting-legged-locomotion-heuristics-rpc](../../wiki/entities/paper-extracting-legged-locomotion-heuristics-rpc.md)
- [paper-bledt-rpc-thesis](../../wiki/entities/paper-bledt-rpc-thesis.md)
- [model-predictive-control](../../wiki/methods/model-predictive-control.md)
