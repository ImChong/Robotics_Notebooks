# FastVLA: Task-Conditioned and Temporally Adaptive Visual Token Compression for Efficient Vision-Language-Action Models

> 来源归档（paper）

- **标题：** FastVLA: Task-Conditioned and Temporally Adaptive Visual Token Compression for Efficient Vision-Language-Action Models
- **中文标题：** FastVLA：面向高效视觉语言动作模型的任务条件与时间自适应视觉 Token 压缩
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.22146>
- **代码：** <https://github.com/jiuyixu25/FastVLA>
- **入库日期：** 2026-10-21
- **线索来源：** [公众号综述（33 篇 Manipulation）](../raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)（具身运控lab，2026-10-20）
- **一句话说明：** 免训练的任务感知、时间自适应视觉 token 压缩，用于降低 VLA 推理开销。
- **沉淀到 wiki：** [`wiki/entities/paper-fastvla-task-adaptive-token-compression.md`](../../wiki/entities/paper-fastvla-task-adaptive-token-compression.md)

## 摘要要点（arXiv 摘要页）

- 高分辨率多视角图像编码出的视觉 token 主导 VLA 的计算与动作延迟。
- 提出免训练的 FastVLA：用任务到视觉的注意力为 token 打分，并按步自适应设定压缩率。

## 开源状态

- **已开源**（2026-10-21 核查）：arXiv 摘要页链接的仓库 `git ls-remote` 可达；README 含 LIBERO 评测与部署脚本。未克隆运行。
