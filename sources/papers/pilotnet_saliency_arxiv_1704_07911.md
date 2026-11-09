# Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car（arXiv:1704.07911）

- **类型：** paper / primary-source reading
- **标题：** Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car
- **作者：** Mariusz Bojarski, Philip Yeres, Anna Choromanska et al.
- **arXiv：** https://arxiv.org/abs/1704.07911
- **HTML：** https://arxiv.org/html/1704.07911v1
- **原文 PDF：** https://arxiv.org/pdf/1704.07911
- **核对范围：** 论文摘要、§1–4 的架构、可视化步骤与显著区域平移消融；参考文献与结论已定位，图像逐幅检查仍待完成。
- **通读：** 未完成
- **一句话摘录：** 对 PilotNet 卷积层特征图逐层平均、上采样并相乘得到显著性掩码，并通过区域平移实验验证输出转向对掩码区域的敏感性。

## 基于当前已核实原文的要点

- 上层激活平均图放大，用作下一层平均激活图的乘性掩码，逐层向输入图像恢复。
- 最终掩码归一化到 [0,1]，叠加在相机图像上。
- 消融：平移显著区域，输出转向变化大；单独平移背景，变化较小。
- 边界：显示相关的决策线索，不等于形式化安全性证明。

## 开放资源核查

- arXiv 论文及 NVIDIA 官方技术博客为一手入口；本轮没有核实到与论文逐项对应的公开可运行实现，不能据此断言永远未开源。
- 补充一手入口：https://developer.nvidia.com/blog/explaining-deep-learning-self-driving-car/

## 对 wiki 的映射

- [paper-pilotnet-saliency.md](../../wiki/entities/paper-pilotnet-saliency.md)
- [11 篇论文路线](../../wiki/overview/robot-learning-foundations-11-papers.md)

## 未完成事项

- 继续阅读尚未核实的正文/附录、表格及可复现资产；完成后方可将“通读”改成“已完成”。
