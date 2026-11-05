# Wall-OSS-0.5 Technical Report（arXiv:2605.30877）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2605.30877>（v1 提交 2026-05-29）
- **PDF：** <https://arxiv.org/pdf/2605.30877>
- **作者：** Ryan Yu, Pushi Zhang, Starrick Liu, Brae Liu, Miracle Kang 等（X Square Robot，arXiv 摘要页前 5 位）
- **代码：** <https://github.com/X-Square-Robot/wall-x>；权重 <https://huggingface.co/x-square-robot/wall-oss-0.5>
- **入库日期：** 2026-11-05
- **线索来源：** LeRobot 固定提交 `9e5ee98` 的 `docs/source/wall_oss_05.mdx`（`policy.type=wall_oss_05`，检查点 `lerobot/wall-oss-0.5`）

## 核心摘录（已核对摘要、引言贡献、结果节中的关键数字）

- 4B 开源 VLA：3B VLM 主干 + 动作专家，MoT 结构，原 3B VLM 保留为 VL Expert。
- 预训练 checkpoint 可直接在真机评测：17 个任务（12 个见过 + 5 个未见）总平均 51.1。
- 微调后 15 个真机任务平均任务进度 60.5%，对 π0.5 的 43.0%（论文表述领先 17.5 个百分点）。
- 报告的训练目标含 flow matching、离散动作 token 交叉熵与多模态交叉熵（「梯度桥」）；推理只走 flow 通路。

## 对 wiki 的映射

- [paper-wall-oss-0-5](../../wiki/entities/paper-wall-oss-0-5.md)
- 一代 WALL-OSS：[cn-os-wall-x](../../wiki/entities/cn-os-wall-x.md)
- LeRobot 实现：[lerobot.md](../repos/lerobot.md)
