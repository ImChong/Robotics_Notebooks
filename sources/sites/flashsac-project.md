# FlashSAC 项目页（holiday-robot.github.io/FlashSAC）

> 来源归档

- **标题：** FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control
- **类型：** site
- **URL：** <https://holiday-robot.github.io/FlashSAC/>
- **论文：** <https://arxiv.org/abs/2604.04539>
- **入库日期：** 2026-07-16
- **一句话说明：** 产品叙事与演示视频：TL;DR「If you're using PPO, try FlashSAC!」；按低/高 DoF、sim-to-real 分组展示 IsaacLab / Genesis / ManiSkill / MuJoCo Playground 与 **Unitree G1** 真机结果。
- **沉淀到 wiki：** [FlashSAC](../../wiki/methods/flashsac.md)

---

## 页面能力要点（策展）

1. **动机叙事：** PPO 在低成本并行仿真下是 sim-to-real 默认，但高维人形/灵巧手/视觉控制中丢弃经验代价过高；off-policy bootstrap critic **慢且不稳**。
2. **三线机制：** (i) 快速训练：少更新 + 大模型 + 高吞吐；(ii) 稳定训练：约束 weight/feature/gradient 范数；(iii) 探索：统一熵目标 + 噪声重复。
3. **视频分组：**
   - **Low DoF：** Anymal-C/D、Go2、Panda/Franka 抓取与 ManiSkill 操作（IsaacLab / Genesis / ManiSkill）。
   - **High DoF：** Shadow/Allegro 立方体重定向、G1/H1 平地/粗糙地形、G1/T1 joystick（IsaacLab / MuJoCo Playground）。
   - **Sim-to-Real (Flat)：** G1 行走、转向、推扰恢复。
   - **Sim-to-Real (Rough)：** G1 楼梯攀爬。
4. **算法细节（页面 Algorithm 区）：** 1024 并行 env、10M replay、2.5M 参数 6 层网络、batch 2048、UTD 2/1024；inverted residual + pre-act BN + RMSNorm；distributional critic；$\sigma_{\mathrm{tgt}}=0.15$ 统一熵；Zeta 分布噪声重复。

## 对 wiki 的映射

- [FlashSAC（方法页）](../../wiki/methods/flashsac.md) — 演示、任务覆盖与工程叙事
- [sources/papers/flashsac_arxiv_2604_04539.md](../papers/flashsac_arxiv_2604_04539.md) — 方法与实验以论文为准
