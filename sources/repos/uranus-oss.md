# D-Robotics Uranus-OSS

> 来源归档

- **标题：** Uranus-OSS
- **类型：** repo（官方推理代码）
- **来源：** 地瓜机器人（D-Robotics）大模型团队
- **链接：** <https://github.com/D-Robotics-AI-Lab/Uranus-OSS>
- **许可证：** Apache-2.0（仓库声明）
- **论文：** [arXiv:2609.24815](../papers/uranus_arxiv_2609_24815.md)
- **项目页：** [官方 Uranus 博客](../sites/d-robotics-uranus.md)
- **演示数据：** <https://huggingface.co/datasets/D-Robotics/Uranus-Demo-Data>
- **模型集合：** <https://huggingface.co/collections/D-Robotics/uranus>
- **核查日期：** 2026-10-09
- **沉淀到 wiki：** [Uranus](../../wiki/entities/paper-uranus.md)

## 仓库能力与使用边界

仓库公开的是论文对应的推理代码和可运行样例；quickstart 通过 `uv` 安装依赖，再调用 `main.py` 并传入权重目录、样例目录和输出目录。样例包括机器人描述、关节轨迹、参考图像和相机信息。若换成自有数据，需准备时间戳对齐的 qpos、机器人 MJCF/URDF、标定好的相机及参考图像。

Hugging Face 集合提供 Uranus-1.3B 与 Uranus-1.3B-Distillation 权重；README 给出的 384×640 推理设置分别使用 25 步和 4 步。数据集是 demo 样例，不应误认成论文训练用的全部 3,300 小时数据；公开推理仓库也不等同于全量训练代码。

## 相关链接

- [论文归档](../papers/uranus_arxiv_2609_24815.md)
- [项目页归档](../sites/d-robotics-uranus.md)
- [项目实体页](../../wiki/entities/paper-uranus.md)
