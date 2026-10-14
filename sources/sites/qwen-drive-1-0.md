# Qwen 官方研究文章：Qwen-Drive-1.0

- **标题：** Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving
- **类型：** site / project-page / official-blog
- **URL：** <https://qwen.ai/blog?id=qwen-drive-1.0>
- **索引页：** <https://qwen.ai/research>
- **配套论文：** [arXiv:2609.00111](https://arxiv.org/abs/2609.00111) — [source 摘录](../papers/qwen_drive_1_0_arxiv_2609_00111.md)
- **代码：** <https://github.com/QwenLM/Qwen-Drive-1.0> — [source 归档](../repos/qwenlm_qwen_drive_1_0.md)
- **权重：** <https://modelscope.cn/models/Qwen/Qwen-Drive-1.0-4B>
- **作者 / 团队：** Qwen Team；论文作者 Xin Zhou 等
- **发布日期：** 2026-09-03
- **核查日期：** 2026-10-14

## 一句话摘要

Qwen 官方文章介绍基于 Qwen3.5-4B 的自动驾驶视觉语言模型，以 BEV 感知头提供可检查 3D 输出，以 flow-matching Planning Expert 生成轨迹，并报告驾驶问答、3D 感知与规划评测。

## 公开信息要点

- **架构：** 共享视觉编码器和 VLM 读取单 / 多视角与时序图像；BEV 头联合做 3D 检测、语义占用和地图分割，规划专家以 flow matching 预测 5 秒轨迹。
- **数据：** 统一跨数据集感知标签、驾驶问答回答和多数据集轨迹；规划数据合计报告为 2.83M 样本，以 10 Hz 预测 5 秒。
- **评测：** 报告驾驶 QA 平均分 69.43；规划涉及 open-loop、pseudo-closed-loop 和 closed-loop 设置，数值须按指标口径阅读。
- **开放状态：** 项目页链接代码、权重和技术报告；代码仓库含推理与评测入口，ModelScope 有 4B 权重页。

## 项目页与代码开放核查

- 项目文章链接 [官方 GitHub](https://github.com/QwenLM/Qwen-Drive-1.0)、[ModelScope 权重](https://modelscope.cn/models/Qwen/Qwen-Drive-1.0-4B) 和论文。
- 仓库可见 scripts/demo.py、scripts/run_vqa.py、scripts/run_planning.py、scripts/run_perception.py 及 NAVSIM / Waymo / PhysicalAI 评测脚本；附 demo 场景数据。
- pyproject.toml 将包描述为 inference code；递归文件树中未见训练脚本。因此判为**部分开源（推理与评测已开源，训练代码未见）**。
- README 标注 Apache-2.0；建议 24 GB 以上 GPU，demo 以 Python 3.10 为例。
- 权重目录约为 VLM 9.1 GB、SFT planner 2.1 GB、RL planner 2.1 GB、perception head 0.5 GB。

## 关联资料

- 论文摘录：[qwen_drive_1_0_arxiv_2609_00111.md](../papers/qwen_drive_1_0_arxiv_2609_00111.md)
- 官方仓库归档：[qwenlm_qwen_drive_1_0.md](../repos/qwenlm_qwen_drive_1_0.md)
- Wiki 项目实体：[paper-qwen-drive-1-0](../../wiki/entities/paper-qwen-drive-1-0.md)
