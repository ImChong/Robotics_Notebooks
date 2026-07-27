# nicklashansen/tdmpc2

> 来源归档

- **标题：** TD-MPC2 官方实现
- **类型：** repo
- **组织 / 作者：** Nicklas Hansen（UCSD）
- **代码：** <https://github.com/nicklashansen/tdmpc2>
- **项目页：** <https://www.tdmpc2.com>
- **论文：** <https://arxiv.org/abs/2310.16828>
- **License：** MIT
- **星标（截至 2026-07-27）：** ~905
- **入库日期：** 2026-07-27
- **一句话说明：** TD-MPC2 训练与评估官方仓：单任务在线 RL + 多任务离线 RL；支持 DMControl / Meta-World / ManiSkill2 / MyoSuite；状态与 RGB 观测；配套 Docker 与大量公开 checkpoint。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `python train.py task=dog-run steps=7000000` | 单任务在线训练 |
| `python train.py task=mt80 model_size=48 batch_size=1024` | 多任务离线训练 |
| `python evaluate.py task=dog-run checkpoint=/path/to/ckpt.pt` | 评估 |
| `python evaluate.py task=mt80 model_size=48 checkpoint=…` | 多任务评估 |
| `obs=rgb` | DMControl 像素观测 |
| `episodic=true` | 终止式任务支持（2025-04 起；默认关以保复现） |
| `docker/` + `environment.yaml` | 环境安装 |

## 资源需求（README）

- 单任务在线：GPU + ≥12 GB RAM；推荐 ≥8 GB 显存
- 多任务 80-task：≥128 GB RAM；317M 训练 ≥24 GB 显存

## 开源状态（仓库核查，2026-07-27）

| 资产 | 状态 |
|------|------|
| 训练 / 评估代码 | **已开源** · MIT |
| Checkpoints / Datasets | 经项目页下载 |
| 任务覆盖 | 104 连续控制任务（论文设定） |

## 对 wiki 的映射

- 论文：[`sources/papers/tdmpc2_arxiv_2310_16828.md`](../papers/tdmpc2_arxiv_2310_16828.md)
- 项目页：[`sources/sites/tdmpc2-com.md`](../sites/tdmpc2-com.md)
- 沉淀 **[`wiki/entities/paper-td-mpc2.md`](../../wiki/entities/paper-td-mpc2.md)**
