# humanoid-parallel-joint-kinematics

> 来源归档（ingest）：人形机器人上**并联 / 闭链关节**的机构学与「解算」相关公开资料索引。

- **入库日期：** 2026-05-12
- **沉淀到 wiki：** 是 → [`wiki/concepts/humanoid-parallel-joint-kinematics.md`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)

## 一句话说明

整理「机构层闭链」与「控制/仿真层等效关节接口」之间的落差，以及教材、开源工具与产业侧踝部并联（如 RSU）叙事中的可引用链接。

## 为什么值得保留

人形下肢常见 **多执行器耦合单任务自由度**（典型为踝 pitch/roll），若只做串联 URDF/MJCF，容易在 **力雅可比、冗余力分配、背隙与弹性** 上产生建模盲区；本索引把分散引用收束到一条 ingest 线，便于后续深挖单篇论文或工具。

## 资料清单（按类型）

### 教材与综述（理论）

| 资料 | 链接 | 备注 |
|------|------|------|
| *Modern Robotics* 第 7 章 *Kinematics of Closed Chains* | 见 [`sources/papers/modern_robotics_textbook.md`](../papers/modern_robotics_textbook.md) 官方 PDF | 闭链结构、Grübler 自由度、并联机构运动学主线 |
| *A Framework for Optimal Ankle Design of Humanoid Robots* | https://arxiv.org/abs/2509.16469 | 人形踝机构设计综述语境，可与 RSU 产品叙事对照 |

### 开源实现（工程）

| 资料 | 链接 | 备注 |
|------|------|------|
| `closed-chain-ik-js` | https://github.com/gkjohnson/closed-chain-ik-js | 浏览器侧闭链 IK 参考实现，适合理解约束迭代与可视化 |

### 产业 / 博客（语境）

| 资料 | 链接 | 备注 |
|------|------|------|
| Menlo / Asimov 下肢叙事：RSU 并联踝 | https://menlo.ai/blog/humanoid-legs-100-days | 强调 **RSU（Revolute–Spherical–Universal）** 与扭矩分担、刚度等产品动机 |

## 对 wiki 的映射

- 升格为概念页 [`wiki/concepts/humanoid-parallel-joint-kinematics.md`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)：归纳「解算」分层、误区与交叉引用。
- 与 [`wiki/entities/asimov-v1.md`](../../wiki/entities/asimov-v1.md) 中 RSU 踝小节互链，避免在实体页重复展开机构公式。
