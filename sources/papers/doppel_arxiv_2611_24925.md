# Doppel｜Render the Robot, Generate the World: One Video World Model Across Embodiments

- **类型：** paper / arXiv:2611.24925
- **原文：** https://arxiv.org/html/2611.24925
- **摘要：** https://arxiv.org/abs/2611.24925
- **项目页：** https://doppelwm.github.io/
- **代码：** https://github.com/DoppelWM/Doppel
- **核查日期：** 2026-11-20
- **核对范围：** arXiv HTML §3-6 渲染条件、相机标定、跨本体训练与结果，长程滚动局限及部分附录；未逐段完成全部附录
- **通读：** 未完成

## 原文记录

1. **方案：** 对动作轨迹已知的本体，用 3D 几何/运动学和相机外参渲染未来机械结构，结合数值动作对 Wan 2.1 视频流匹配模型加条件。
2. **条件通道：** 未来 render 用冻结 VAE 编码，动作按本体归一化并零补齐 159 维，渲染–动作 adapter 融合并以 per-token timestep modulation 注入去噪网络。
3. **数据：** 8 个数据集、8 类机器人及人体本体，约 600 h；缺乏外参的数据通过视频拟合校准。Doppel-8 没有输入 embodiment label，但评测本体都曾见于训练。
4. **指标：** 三个机械臂数据集较 action-only baseline，未见轨迹 +1.00 dB PSNR，OOD +1.20 dB；另报告多本体条件训练可节约训练抽样次数 63.8%，不可混为时延加速。
5. **局限：** 25.6 s 只是串接八个 3.2 s 片段的定性轨迹演示，长程误差会积累；不能等同策略成功率。
6. **许可：** 代码 MIT；**权重 CC BY-NC 4.0**，Wan 基座依作者条款；不能将项目 MIT 扩展到权重商用。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-doppel.md)
