# TurboVLA 官方源码导航（2026-11-14）

- **资料类型：** repo / 论文对应开源实现
- **名称：** TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM
- **中文：** TurboVLA：面向实时推理的轻量视觉语言动作模型
- **论文：** https://arxiv.org/abs/2607.27205
- **项目页：** https://h-embodvis.github.io/TurboVLA/
- **代码：** https://github.com/H-EmbodVis/TurboVLA
- **权重：** https://huggingface.co/H-EmbodVis/TurboVLA
- **机构：** 华中科技大学、华为技术有限公司
- **核对范围：** 官方项目页与仓库 README 的摘要、版本动态、Architecture、Performance、Installation、Dataset/Model Preparation、Training and Evaluation 章节。
- **通读：** 未完成（已核对 README 关键章节，未通读论文全文和全部训练脚本）。

## 机制与开放边界

直接采用视觉特征与文本特征交互后生成连续动作块，而非让大语言模型在每个控制时刻作为必经中枢。官方 README 给出的 RTX 4090 上 0.2B 模型 LIBERO 约 97.6% 平均成功率、31.2 ms 延迟和约 0.9 GiB 推理显存；0.4B 模型 RoboTwin 2.0 约 88.06%（2026-09-27 更新）。**成功率属于不同模型/基准，不能拼接比较。**

代码与权重：**已开放**。README 分别提供 LIBERO/RoboTwin 安装环境；训练入口 `experiments/libero/train.py`，LIBERO 评测入口 `experiments/libero/evaluate.py`，RoboTwin 数据准备 `scripts/robotwin/prepare_data.sh`。仅按 README 静态核对，**未在 GPU 上执行**。

## 对 wiki 的映射

- [TurboVLA 独立论文详情](../../wiki/entities/paper-turbovla.md)
- [2026 VLA 开源生态对比](../../wiki/comparisons/vla-open-source-ecosystem-2026-three-shifts.md)
- [VLA 方法](../../wiki/methods/vla.md)

## 参考来源

- https://github.com/H-EmbodVis/TurboVLA
- https://h-embodvis.github.io/TurboVLA/
- https://arxiv.org/abs/2607.27205
