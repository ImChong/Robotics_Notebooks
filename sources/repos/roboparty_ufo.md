# UFO（Roboparty 无监督 RL 控制框架）

> 来源归档

- **标题：** UFO — Unsupervised RL Control Development Framework
- **类型：** repo（无监督强化学习控制框架）
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/UFO
- **入库日期：** 2026-07-14
- **一句话说明：** 面向研发、完全开源的人形机器人无监督强化学习控制开发框架，覆盖训练基础设施、数据管线、算法研究与推理部署全流程。
- **沉淀到 wiki：** 是（`wiki/entities/roboparty-ufo.md`）

---

## 核心能力（文内自述）

| 能力 | 说明 |
|------|------|
| Fast Training | MJLab backend；8×4090 <12h 完成 BFM-Zero；8×H200 6–8h |
| 通用可扩展 | 无缝适配不同机器人形态；多来源数据混合训练与调度 |
| 表征集成 | BFM-Zero（FB Representation）；探索 TeCH 等新表征 |
| 真机遥操 | 首次开源无监督 RL 控制遥操作代码与完整验证方案 |

---

## 命名辨析

- **UFO（本仓库）** = Roboparty 无监督 RL 控制框架，与通用缩写无关
- 与 [paper-bfm-zero](../../wiki/entities/paper-bfm-zero.md) 的关系：UFO 集成并加速 BFM-Zero 类算法训练

---

## 对 wiki 的映射

- [roboparty-ufo](../../wiki/entities/roboparty-ufo.md)
- 父级：[party-os](../../wiki/entities/party-os.md)
- 交叉：[mjlab](../../wiki/entities/mjlab.md)、[paper-bfm-zero](../../wiki/entities/paper-bfm-zero.md)
