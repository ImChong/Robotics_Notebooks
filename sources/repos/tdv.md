# ninaddaithankar/TDV

> 来源归档（ingest）

- **仓库：** <https://github.com/ninaddaithankar/TDV>
- **对应论文：** [TDV（arXiv:2606.15956）](../papers/tdv_arxiv_2606_15956.md)
- **许可证：** Apache-2.0
- **环境（README）：** Python 3.11 conda 环境，PyTorch（CUDA 12.6），`requirements.txt` 以 `--no-deps` 安装；需设置 `HF_HOME` 并登录 wandb
- **训练（README）：** 编辑 `job_scripts/pretrain_tdv.slurm` 后执行；支持 `--debug_mode`、`--no_wandb`；集群提交见 `slurm_executor.sh`
- **评测（README）：** 线性探针（SSv2 / ImageNet）、光流（Sintel）、立体深度（SceneFlow）、语义分割（ADE20K / Cityscapes，mmsegmentation fork）；需配置 checkpoint 与数据路径
- **数据说明：** `data/cv/README.md`
- **入库日期：** 2026-10-20

## 对 wiki 的映射

- [paper-tdv](../../wiki/entities/paper-tdv.md) 的「源码运行时序图」与「工程实践」
