# redorangeyellowy/EgoWorld

> 来源归档

- **标题：** EgoWorld（官方 PyTorch 实现，ICLR 2026）
- **类型：** repo
- **组织 / 作者：** redorangeyellowy（Junho Park 等）
- **代码：** <https://github.com/redorangeyellowy/EgoWorld>
- **论文：** <https://arxiv.org/abs/2506.17896>
- **项目页：** <https://redorangeyellowy.github.io/EgoWorld/>
- **许可：** MIT
- **入库日期：** 2026-07-24
- **一句话说明：** ControlNet/LDM 风格 inpainting 训练与测试入口；对齐 README 的 H2O root、SD-inpainting 预训练与 H2O checkpoint；核心目录 `cldm/`、`ldm/`、`models/inpainting.yaml`、`datasets.py`。

## 入口速查（对齐 README · 2026-07-24）

| 路径 / 命令 | 作用 |
|-------------|------|
| `pip install -r requirements.txt` | 安装依赖 |
| `python3 train.py --root_path … --pretrained_path …` | 训练；H2O root 默认 `./database/h2o`；预训练默认 SD-v1-5-inpainting ckpt |
| `python3 test.py --root_path … --pretrained_path …` | 评测；H2O 预训练 ckpt 示例路径见 README |
| `models/inpainting.yaml` | ControlNet/LDM inpainting 配置 |
| `cldm/` · `ldm/` | 扩散 / ControlNet 实现 |
| `datasets.py` · `data/h2o_action/inpainting/*.json` | 数据加载与 H2O action split |

## 项目页 / 源码开放核查（步骤 2.5）

- **状态：已开源**（代码 + README 可运行入口 + 公开权重/预处理下载链接）。
- **边界：** 深度估计、exo MANO 手姿、VLM 文本仍为论文中的 off-the-shelf 组件，仓库聚焦扩散重建阶段；TACO 等额外数据集需自行对齐 root。

## 与本仓库知识的关系

- 论文归档：[`sources/papers/egoworld_arxiv_2506_17896.md`](../papers/egoworld_arxiv_2506_17896.md)
- 项目页：[`sources/sites/egoworld-github-io.md`](../sites/egoworld-github-io.md)
- wiki：[`wiki/entities/paper-egoworld.md`](../../wiki/entities/paper-egoworld.md)
- **消歧：** 非 [EgoWorld-100W](../blogs/stellarnex_egoworld_100w.md) 数据集仓
