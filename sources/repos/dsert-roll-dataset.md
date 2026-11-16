# jeongyh98/DSERT-RoLL-Dataset 数据集与可视化

- **官方仓库：** https://github.com/jeongyh98/DSERT-RoLL-Dataset
- **数据卡：** https://huggingface.co/datasets/jeongyh98/DSERT-RoLL
- **项目页：** https://jeongyh98.github.io/dsert-roll/
- **论文：** https://arxiv.org/abs/2604.03685
- **开放状态：** README 发布数据下载链接、目录格式与 `viz.py` 可视化工具。报告总量约 **269 GB**，132 train + 58 val sequences。
- **主要内容：** `split/train.txt`、`split/val.txt`、按天气/sequence 的 `label.pkl`；RGB/Event/Thermal 左右目、Livox/Ouster/4D Radar 帧、标定及 3D 框与 object ID。
- **代码入口：** `python viz.py --root_path /path/to/DSERT-RoLL --bbox`。工具生成按 RGB/Event/Thermal 行、LiDAR/Ouster/Radar × 左右目列的 3×6 组合图，可选叠加框。
- **重要提醒：** 数据仓库内的 `viz.py` 是可视化，不是论文所有检测算法的训练实现。3D 检测另见 [3DOD](./dsert-roll-3dod.md)，双目另见 [LIGA](./dsert-roll-liga-stereo.md)。
- **许可：** 论文补充材料 §1 指定研究用途 CC BY-NC 4.0；实际使用还应核对数据卡及发布条款。
- **主节点：** [DSERT-RoLL](../../wiki/entities/paper-dsert-roll.md)
