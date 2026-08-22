# PVRA（KulunuOS/PVRA）

- **URL：** <https://github.com/KulunuOS/PVRA>
- **论文：** [arXiv:2608.19968](https://arxiv.org/abs/2608.19968)
- **许可：** MIT

## 入口

- 环境：`mamba env create -f environment.yml`
- 预处理：`tools/preprocess_cld_rgb_nrms.py`
- 训练：`python -m train.train_L_multi_dataset`
- 推理：`python -m eval.inference`

## 数据

- 外部 [6DAPose / Nema17](https://zenodo.org/records/10117869) 需自行下载。

## wiki

- [`wiki/entities/paper-pvra.md`](../../wiki/entities/paper-pvra.md)
