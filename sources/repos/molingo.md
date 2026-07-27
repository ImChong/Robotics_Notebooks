# hynann/MoLingo

- **URL（仓库）：** <https://github.com/hynann/MoLingo>
- **URL（项目页）：** <https://hynann.github.io/molingo/MoLingo.html> — 归档见 [`molingo-github-io.md`](../sites/molingo-github-io.md)
- **维护方：** Yannan He 等（University of Tübingen / MPI-INF / Imperial College London）
- **定位：** **MoLingo（CVPR 2026）** 官方代码：语义对齐连续潜空间上的 **掩码自回归 rectified-flow** 文本→人体运动生成
- **论文：** arXiv:2512.13840 — 归档见 [`molingo_arxiv_2512_13840.md`](../papers/molingo_arxiv_2512_13840.md)
- **许可：** Apache-2.0
- **入库日期：** 2026-07-27

## 仓库要点（维护者速览）

| 项 | 内容 |
|----|------|
| **环境** | `conda env create -f environment.yml`；Python 3.10.13 / PyTorch 2.9.0 / CUDA 12.8（README） |
| **数据** | HumanML3D 263D（[EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D)）；HumanML3D-272（[MotionStreamer](https://github.com/zju3dv/MotionStreamer)）；SAE 需 BABEL 帧级特征包 |
| **权重** | `prepare/download_models.sh`；评测器 `download_evaluator.sh` + glove；可选 TMR-263 |
| **训练 SAE** | `python mogen/train_sae.py --data_root {data_root}` → `mogen/checkpoints/ms/{vae_name}` |
| **训练生成** | `torchrun … mogen/train_molingo.py --data_root … --vae {vae_name}`；4×A100/H100 级 |
| **Demo** | `python mogen/demo.py -a 1 -i assets/example.txt -b {smpl_path}`（**仅 272D 模型**入口；单卡 3090） |
| **评测** | `mogen/eval_mogen.py -d 263|272 …`（TMR-263 / MARDM-67 / MS-272） |
| **未发布** | README TODO：`Release the G1 tracking pipeline`（项目页有 G1 视频，代码侧跟踪管线仍待放） |

## 推荐复现路径（最短）

1. 建 conda 环境 → `prepare/download_models.sh`
2. 准备 SMPL 权重 → `mogen/demo.py` 用 `assets/example.txt` 生成（推荐 272D checkpoint）
3. 若要重训 / 刷表：准备 HumanML3D（±272）与评测器，再跑 `train_sae` → `train_molingo` → `eval_mogen`

## 对 wiki 的映射

- 论文实体：[paper-molingo](../../wiki/entities/paper-molingo.md)
- 论文摘录：[molingo_arxiv_2512_13840.md](../papers/molingo_arxiv_2512_13840.md)
- 项目页：[molingo-github-io.md](../sites/molingo-github-io.md)
