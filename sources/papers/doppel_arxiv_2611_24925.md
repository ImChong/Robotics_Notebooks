# Doppel｜Render the Robot, Generate the World: One Video World Model Across Embodiments

- **类型：** paper / arXiv:2611.24925
- **版本：** arXiv:2611.24925v1，2026-11-18
- **作者：** Yash Jangir、Karan Mirakhor、Renos Zabounidis、Muhammad Zubair Irshad、Sergey Zakharov、Yonatan Bisk、Homanga Bharadhwaj
- **机构：** 约翰斯·霍普金斯大学、卡内基梅隆大学、丰田研究院
- **原文：** [HTML](https://arxiv.org/html/2611.24925v1) · [PDF](https://arxiv.org/pdf/2611.24925v1)
- **摘要：** https://arxiv.org/abs/2611.24925
- **项目页：** https://doppelwm.github.io/
- **代码：** https://github.com/DoppelWM/Doppel
- **核查日期：** 2026-11-26
- **核对范围：** 45 页 v1 PDF 的 §1–7、参考文献与附录 A.1–A.10；核对不同训练预算下的表 2/10/12/13、相机恢复与发布文件、长时序和部署边界。
- **通读：** 已完成

## 原文记录

1. **方案：** 对动作轨迹已知的本体，用 3D 几何/运动学和相机外参渲染未来机械结构，结合数值动作对 Wan 2.1 视频流匹配模型加条件。
2. **条件通道：** 未来 render 用冻结 VAE 编码，动作按本体归一化并零补齐 159 维，渲染–动作 adapter 融合并以 per-token timestep modulation 注入去噪网络。
3. **数据：** 8 个数据集、8 类机器人及人体本体，约 600 h；缺乏外参的数据通过视频拟合校准。Doppel-8 没有输入 embodiment label，但评测本体都曾见于训练。
4. **指标：** 三个机械臂数据集较 action-only baseline，未见轨迹 +1.00 dB PSNR，OOD +1.20 dB；另报告多本体条件训练可节约训练抽样次数 63.8%，不可混为时延加速。
5. **局限：** 25.6 s 只是串接八个 3.2 s 片段的定性轨迹演示，长程误差会积累；不能等同策略成功率。
6. **许可：** 代码 MIT；**权重 CC BY-NC 4.0**，Wan 基座依作者条款；不能将项目 MIT 扩展到权重商用。

**附录核查补充：** A.4 中稀疏本体渲染的 VAE 特征先按通道做时空均值中心化，否则稀疏机器人信号可能被空背景淹没；适配器约 30.5M 参数，通过两个 cross-attention block 把动作融入 render tokens。A.7 表 9：803,602 训练 clip、593 h，Doppel-8 使用 32× H100 训练；所有被评测的本体均已参与混训，不是未见机器人零样本迁移。A.8 评测 PSNR / SSIM / LPIPS + robot 与运动物体区域指标，16 个预测帧中不计第一条件帧。A.10 表 10 特别指出 Bridge OOD 上 Doppel-8 比 Doppel-3 低 0.24 dB，跨本体混训不保证每个分组都变好。推理依赖准确外参和预渲染，不应将视觉生成保真度直接等同物理接触可执行性。**代码 MIT；权重 CC BY-NC 4.0。**

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-doppel.md)
