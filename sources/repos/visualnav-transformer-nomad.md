# NoMaD（robodhruv/visualnav-transformer）

> 来源归档

- **标题：** General Navigation Models: GNM, ViNT and NoMaD
- **类型：** repo
- **来源：** UC Berkeley
- **链接：** <https://github.com/robodhruv/visualnav-transformer>
- **项目页：** <https://general-navigation-models.github.io/nomad/>
- **论文：** <https://arxiv.org/abs/2310.07896>
- **许可：** MIT
- **入库日期：** 2026-07-28
- **一句话说明：** NoMaD / ViNT / GNM 的训练、预训练 checkpoint 与 TurtleBot2 / LoCoBot ROS 部署实现。
- **开源状态：** **已开源**；公开数据可重训，但论文使用的部分 Seattle / SCAND 轨迹未公开。
- **沉淀到 wiki：** [`paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md`](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 训练 | `train/train.py -c train/config/nomad.yaml` |
| 数据处理 | `train/process_*.py` → images + `traj_data.pkl` |
| 模型配置 | `train/config/nomad.yaml`（10 diffusion iters、goal mask 0.5） |
| 拓扑图 | `deployment/src/record_bag.sh`、`create_topomap.sh` |
| 目标导航 | `deployment/src/navigate.sh` |
| 探索 | `deployment/src/explore.sh`（NoMaD only） |
| 环境 | ROS Noetic；Ubuntu 18.04 / 20.04；Python 3.7+；CUDA 10+ |

## 对 wiki 的映射

- [NoMaD 论文实体](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md)
- [NoMaD 项目页](../sites/nomad.md)
- [论文 source](../papers/humanoid_pnb_nomad-goal-masked-diffusion-policies-for-navigat.md)
