# Gymnasium LunarLander 环境文档

> 来源归档（ingest）

- **页面：** <https://gymnasium.farama.org/environments/box2d/lunar_lander/>
- **类型：** 官方文档（Farama Foundation）
- **入库日期：** 2026-11-06
- **核对范围：** 文档页全文（当前默认 ID 为 `LunarLander-v3`）。
- **通读：** 已完成

## 核心摘录

1. **动作。** 默认 `Discrete(4)`：不动作 / 左姿控发动机 / 主发动机 / 右姿控发动机；`continuous=True` 时为 `Box(-1, +1, (2,))`（主推力、横向推力），主推力 main < 0 时关机，0–1 对应 50%–100% 推力。
2. **观测。** 8 维：x、y、vx、vy、角度、角速度、两条腿是否触地。
3. **奖励。** 距离、速度、倾角塑形，每条腿触地 +10，侧发动机每帧 −0.03、主发动机每帧 −0.3，坠毁 −100、安全着陆 +100；"An episode is considered a solution if it scores at least 200 points."
4. **离散动作的理由。** "According to Pontryagin's maximum principle, it is optimal to fire the engine at full throttle or turn it off."
5. **可选扰动。** `enable_wind`、`wind_power`（0–20）、`turbulence_power`（0–2），重力默认 −10（可设 0 到 −12）。

## 对 wiki 的映射

- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 6 入门练习（平面、无质量变化，只适合熟悉 RL 接口）
