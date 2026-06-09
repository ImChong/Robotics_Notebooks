# DIMOS

> 来源归档

- **标题：** DIMOS
- **类型：** repo
- **来源：** Kaifeng Zhao（zkf1997）/ ETH Zürich
- **链接：** <https://github.com/zkf1997/DIMOS>
- **Stars：** ~104（2026-06）
- **入库日期：** 2026-06-09
- **一句话说明：** ICCV 2023 **DIMOS** 官方实现：RL 驱动的室内 human-scene interaction 运动合成（locomotion + 坐/躺/站起），基于 GAMMA 运动基元栈与 COINS 静态交互目标。
- **沉淀到 wiki：** [`wiki/entities/paper-dimos-human-scene-motion-synthesis.md`](../../wiki/entities/paper-dimos-human-scene-motion-synthesis.md)

---

## 核心定位

开源 **复杂室内 3D 场景中多样人体运动合成** 复现入口：在 **GAMMA** 的 CVAE 运动基元 + PPO 控制框架上扩展 **场景可走性图**、**物体 SDF 交互特征** 与 **COINS marker 目标**，配合 NavMesh 路点实现长程 **走–坐–走–躺** 等活动序列。

## 依赖栈

| 组件 | 说明 |
|------|------|
| OS / CUDA | Ubuntu 22.04, CUDA 11.8（README 实测） |
| Conda | `conda env create -f env.yml` |
| 人体模型 | SMPL-X, VPoser v1.0 |
| 动捕数据 | AMASS, SAMP, BABEL（训练）；demo 可部分省略 |
| 场景/物体 | ShapeNet, Replica, PROX-S（训练/评测） |
| Marker | CMU 41 / SSM2 67 marker placement |

## 仓库要点

- **许可：** 自研部分 Apache 2.0；SMPL-X / AMASS 等第三方数据遵循各自许可
- **更新：** 运动生成 demo、数据预处理与训练文档；提供清洗版 locomotion 权重（Google Drive）
- **上游：** 基于 [GAMMA](https://github.com/yz-cnsdq-z/GAMMA) 代码栈；静态交互目标依赖 [COINS](https://github.com/zkf1997/COINS)

## 对 wiki 的映射

- 论文归档：[sources/papers/dimos_arxiv_2305_12411.md](../papers/dimos_arxiv_2305_12411.md)
- 项目页：[sources/sites/dimos-zkf1997-github-io.md](../sites/dimos-zkf1997-github-io.md)
- 实体页：[wiki/entities/paper-dimos-human-scene-motion-synthesis.md](../../wiki/entities/paper-dimos-human-scene-motion-synthesis.md)
