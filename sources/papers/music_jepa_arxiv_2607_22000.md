# Music-JEPA: Learning a World Model of Piano Sound from Action（arXiv:2607.22000）

> 来源归档（ingest）

- **标题：** Music-JEPA: Learning a World Model of Piano Sound from Action
- **类型：** paper / JEPA / audio world model / planning
- **arXiv：** <https://arxiv.org/abs/2607.22000>
- **项目页：** <https://zzwaang.github.io/music-jepa-demo/>
- **代码：** <https://github.com/ZZWaang/music-jepa>（原创代码 MIT；第三方组件保留各自许可证）
- **数据 / 权重：** MAESTRO 衍生数据集与预训练权重托管于 Hugging Face（README 所述）
- **作者：** Ziyu Wang；Kun Fang；Yann LeCun
- **机构（项目页）：** New York University；MBZUAI；McGill University；CIRMMT；AMI Labs
- **入库日期：** 2026-10-20
- **来源文章：** [wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md](../blogs/wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md)

## 开源核查（步骤 2.5，2026-10-20）

**部分开源**：训练脚本（`scripts/train_jepa.py`）、数据与预训练权重已公开；README 明确下游实验代码「将在后续更新中加入」，规划 / 评测入口未给出。详见 [paper-music-jepa](../../wiki/entities/paper-music-jepa.md) 的「工程实践」。

## 摘录要点（归纳）

- 把当前声音作为状态、按键与踏板作为动作、动作后的声音作为下一状态，在潜空间学习 action-conditioned dynamics。
- 文章转述：表示可用于节拍检测、作曲家与调性识别；模型可反向用于规划，从目标音乐推断按键序列（钢琴转录）。

## 对 wiki 的映射

- 主节点：[paper-music-jepa](../../wiki/entities/paper-music-jepa.md)
