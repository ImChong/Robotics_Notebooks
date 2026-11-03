# Ψ-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model / Ψ-0.6：统一视觉语言世界模型中的自适应思维流

- **论文：** <https://arxiv.org/abs/2610.32177>（v1，2026-10-23）
- **原文：** <https://arxiv.org/html/2610.32177v1>
- **作者：** Khai Loong Aw, Klemen Kotar, Baihan Zhang, Yifan Liu, Lilian Chen, Atlas Kazemian, Rahul Venkatesh, Daniel Yamins
- **机构：** 斯坦福大学
- **项目页：** <https://neuroailab.github.io/psi-website/blog/psi-0-6.html>
- **代码入口：** <https://github.com/neuroailab/psi-0.6>
- **入库日期：** 2026-11-03

## 已核实的线索

- 论文描述一个约 8B 参数的自回归模型，以共享 token 接口处理图像、文本、相机位姿、深度、光流和点轨迹；预测结果可作为后续推理的上下文。
- 基于视频与问题选择中间表示，使用答案损失作为强化学习信号；论文附录 C.3 描述初始化和后续策略优化。
- 论文表 3 中 FlowBench-Real 的 Ψ-0.6 经 FlowBench-Sim 微调后总体准确率为 51.4%；该结果是问答迁移，不等同于机器人控制。

## 待完成

已检查摘要、正文与附录的部分段落；**尚未通读完整论文**。须继续阅读全文、核对实验和局限后，编写项目实体页与基于原文的 Mermaid 流程图。官方仓库尚无可运行实现，暂不绘制源码时序图。

## 相关归档

- [项目页](../sites/psi-0-6.md)
- [官方仓库](../repos/psi-0-6.md)
