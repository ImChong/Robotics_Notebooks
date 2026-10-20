# ZZWaang/music-jepa

> 来源归档（ingest）

- **仓库：** <https://github.com/ZZWaang/music-jepa>
- **对应论文：** [Music-JEPA（arXiv:2607.22000）](../papers/music_jepa_arxiv_2607_22000.md)
- **许可证：** 原创代码 MIT；第三方组件保留各自许可证（MIT、LGPL-3.0、CC BY-NC 4.0、Apache-2.0）
- **环境（README）：** Python 3.10，`pip install -r requirements.txt`
- **数据（README）：** MAESTRO 衍生数据集（1,276 段演奏，约 200 小时），托管于 Hugging Face，下载到 `data/maestro/`
- **权重（README）：** 预训练权重托管于 Hugging Face，下载到 `checkpoints/`
- **入口（README）：** 训练 `scripts/train_jepa.py` + 配置文件；README 未列推理、规划与评测入口，称「下游实验代码将在后续更新中加入」
- **入库日期：** 2026-10-20

## 对 wiki 的映射

- [paper-music-jepa](../../wiki/entities/paper-music-jepa.md) 的「源码运行时序图」与「工程实践」
