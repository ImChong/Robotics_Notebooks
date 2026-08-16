# Gymnasium Cart Pole（CartPole-v1）官方环境页

- **标题：** Cart Pole — Gymnasium Documentation
- **类型：** site / 官方文档
- **URL：** <https://gymnasium.farama.org/environments/classic_control/cart_pole/>
- **源码：** <https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py>
- **动力学注释引用：** Florian, *Correct equations for the dynamics of the cart-pole system*（2005/2007）— <https://coneural.org/florian/papers/05_cart_pole.pdf>
- **配套仓库归档：** [`sources/repos/gymnasium.md`](../repos/gymnasium.md)
- **入库日期：** 2026-08-16
- **一句话说明：** Farama 对 Barto–Sutton–Anderson 1983 cart-pole 的标准 Python 实现与 API 契约；是 RL 入门与算法对照的默认离散动作基准。
- **代码：** 已开源（MIT）→ [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- **沉淀到 wiki：** 是 → [`wiki/concepts/cartpole.md`](../../wiki/concepts/cartpole.md)

---

## 官方契约（文档 + 源码核对，2026-08-16）

| 项 | 官方值 |
|----|--------|
| 注册 id | `gymnasium.make("CartPole-v1")`（v0 的步数上限 200；v1 为 500） |
| 动作空间 | `Discrete(2)`：0 向左推、1 向右推；固定力幅 `force_mag = 10.0` N |
| 观测空间 | `Box(4,)`：小车位置、小车速度、杆角、杆角速度 |
| 观测盒边界 | 位置 ±4.8、杆角 ±24°（约 0.418 rad）；速度无界 |
| 终止（terminated） | \|杆角\| > 12°（0.2095 rad）或 \|小车位置\| > 2.4 m |
| 截断（truncated） | v1：500 步；v0：200 步（`TimeLimit` wrapper） |
| 默认奖励 | 每步 +1（含终止步）；solved 阈值 500（v1）/ 200（v0） |
| `sutton_barto_reward=True` | 未终止 0、终止 −1；阈值改为 0（对齐 1983 失败信号） |
| 初始状态 | 四维均匀随机 `(-0.05, 0.05)`；`reset(options={low, high})` 可改 |
| 物理参数 | \(g=9.8\)，\(m_c=1.0\)，\(m_p=0.1\)，杆半长 \(l=0.5\)，\(\tau=0.02\) s（50 Hz） |
| 积分 | 默认 Euler；源码另有 semi-implicit Euler 分支 |
| 向量化 | `make_vec("CartPole-v1", ...)`；专用 `CartPoleVectorEnv` |

文档强调：**观测空间盒 ≠ 未终止允许区间**。位置可观测到 ±4.8，但 ±2.4 已终止；杆角可观测到 ±24°，但 ±12° 已终止。

## 开源核查（步骤 2.5）

- 文档页本身不是项目营销页，但是 **Farama 官方环境规范**；源码与文档同仓。
- **已开源、可运行**：`pip install gymnasium` 后 `gym.make("CartPole-v1")` 即可；classic-control 渲染需 pygame extra。

## 对 wiki 的映射

- [Cartpole 问题](../../wiki/concepts/cartpole.md)
- [Gymnasium](../../wiki/entities/gymnasium.md)
- [具身 RL 最小闭环](../../wiki/concepts/embodied-rl-minimal-closed-loop.md)
- [Reward Design](../../wiki/concepts/reward-design.md)

## 为什么值得保留

- 这是把 1983 论文落成 **可 `make` 的 MDP 契约** 的一手规范：动作、观测、终止/截断、两种奖励，缺一不可。
- 与 Isaac Lab `Isaac-Cartpole-v0` 对照时，必须用本页数字，而不是凭记忆把「CartPole」当成同一个环境。
