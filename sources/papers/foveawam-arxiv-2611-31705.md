# FoveaWAM：论文及补充材料归档 | Paper and supplement

- **类型：** paper
- **论文：** [arXiv:2611.31705v1](https://arxiv.org/abs/2611.31705v1) · [HTML](https://arxiv.org/html/2611.31705v1) · [PDF](https://arxiv.org/pdf/2611.31705v1)
- **作者：** Jiaqi Yin、Baoyue Zhang、Jiyuan Zhang、Wenxuan Liu、Zhaofei Yu、Yue Deng、Tiejun Huang
- **机构：** 北京中关村学院、北京大学、北京智源人工智能研究院
- **项目页：** <https://foveawam.github.io/>
- **代码：** <https://github.com/ArrebolBlack/FoveaWAM>
- **核对范围：** arXiv v1 正文 §1–6 与补充 A–D，核对预测监督、部署路径、执行器、基线与真机评测。
- **通读：** 已完成
- **入库日期：** 2026-11-25
- **对应节点：** [FoveaWAM](../../wiki/entities/paper-foveawam.md)

1. §3 / 附录 A：当前 RGB-D 与位姿/机器人状态历史进入模型；训练通过目标 SE(3) 和周围稀疏 3D 流监督世界流，动作流用 flow matching。部署可跳过两套世界预测头，但仍计算世界流条件并缓存 K/V。
2. §4 / 附录 B：OnTimeBench Native 依据各方法在 RTX 5090 测得的延迟推进物理时间，等待时执行旧动作或保持，返回后略过已过期的新动作。Paused 不是同一个时序协议。
3. §5 / 表 1、4 与附录 D：Native 82.1%（PUMA 61.4%），p95 首命令 15.1 ms。六项真机均 20 次，FoveaWAM 48.3% 对 π0.5 30.8%；真机从就绪图像到完整 CPU 动作块、包含位姿追踪为 p95 25.17 ms，两种计时不能代换。

[项目页归档](../sites/foveawam.md) · [源码与复现归档](../repos/foveawam.md)
