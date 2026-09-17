# RecMorph

> 来源归档

- **标题：** RecMorph
- **类型：** repo
- **链接：** <https://github.com/quanruirao/RecMorph>
- **论文：** <https://arxiv.org/abs/2609.18359>
- **入库日期：** 2026-09-17
- **一句话说明：** RecMorph 官方实现：UNIMAL 广义形态 MuJoCo + Isaac Lab 四足跨平台 locomotion。
- **代码：** **已开源**
- **沉淀到 wiki：** [`paper-recmorph`](../../wiki/entities/paper-recmorph.md)
- **交叉归档：** [`recmorph_arxiv_2609_18359.md`](../papers/recmorph_arxiv_2609_18359.md)

## 仓库布局

| 路径 | 说明 |
|------|------|
| `unimal/` | MuJoCo/UNIMAL：metamorph 环境、PPO、BiRNN/BiLSTM/BiGRU/BiMamba2 |
| `isaaclab/recmorph_locomotion/` | Go1/Go2/ANYmal-B/C 共享环境与策略 |
| `scripts/` | 可移植 train/eval launcher |
| `docs/UNIMAL.md` | 复现协议 |

## 复现要点

- UNIMAL：`conda env create -f environment/unimal.yml` → `pip install -e unimal` → `bash scripts/download_unimal_data.sh`
- 训练示例：`bash scripts/train_unimal.sh ft birnn 1409`
- Isaac Lab：目标 Isaac Lab 0.41.3 + Isaac Sim 4.5 + RSL-RL 2.3.1
