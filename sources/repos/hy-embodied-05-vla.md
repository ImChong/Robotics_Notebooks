# Hy-Embodied-0.5-VLA 官方源码导航（2026-11-14）

- **资料类型：** repo / VLA 训练到部署系统
- **英文标题：** From Vision-Language-Action Models to a Real-World Robot Learning Stack
- **中文标题：** 从 VLA 模型到真实机器人学习全栈
- **团队：** Tencent Robotics X × Tencent Hy Team
- **项目页：** https://tairos.tencent.com/openSourceModels/hy-embodied-0.5-vla
- **论文：** https://arxiv.org/abs/2606.14409
- **代码：** https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA
- **权重：** https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI
- **公开数据：** https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data
- **核对范围：** 官方 README 的更新日志、模型结构、数据/权重资源、Quick Start 和训练脚本目录。
- **通读：** 未完成（已核对 README 与开放入口，未通读论文全文及全部代码）。

## 机制与开放边界

采用 HY-Embodied-0.5 MoT 视觉–语言骨干 + 双塔 flow matching 动作专家、时空记忆编码和相对末端 delta-chunk 动作表示。README 说明训练规模 **10,000+ 小时 UMI**，但 2026-06-15 公开发布的 UMI 数据是 **2,000+ 小时**，二者不得混用。

代码、RoboTwin/UMI 权重、部分数据：**已开放**。可识别运行入口为 `python scripts/quick_start.py`、`scripts/train_umi_vlm.sh`、`scripts/eval_robotwin_full.sh`；README 给出 `HyVLA.from_pretrained` 与 `forward_evaluate` 的推理示例。未运行或验证真实控制时序。

## 对 wiki 的映射

- [2026 VLA 开源生态对比](../../wiki/comparisons/vla-open-source-ecosystem-2026-three-shifts.md)
- [HY-Embodied 家族现有入口](../../wiki/entities/cn-os-hy-embodied.md)
- [已收录公开数据集](../../wiki/entities/dataset-hy-embodied-0-5-vla-data.md)

## 参考来源

- https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA
- https://tairos.tencent.com/openSourceModels/hy-embodied-0.5-vla
- https://arxiv.org/abs/2606.14409
