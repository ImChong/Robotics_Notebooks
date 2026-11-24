# DecomVoxel: Harnessing 3D-Native Priors with Guided In-situ Denoising Optimization for Decompositional Scene Reconstruction

- **论文：** https://arxiv.org/abs/2610.01914
- **项目：** https://decomvoxel.github.io/DecomVoxel-Webpage/
- **代码：** https://github.com/DecomVoxel/DecomVoxel
- **数据：** https://huggingface.co/datasets/JunfengNi/DecomVoxel
- **版本：** arXiv:2610.01914v1，2026-10-01；SIGGRAPH Asia 2026 Journal Track（ACM TOG），DOI: 10.1145/3842522
- **全文：** https://arxiv.org/html/2610.01914v1
- **PDF：** https://arxiv.org/pdf/2610.01914
- **作者：** Junfeng Ni、Zirui Zhou、Yixin Chen、Yu Liu、Nan Jiang、Zhifei Yang、Song-Chun Zhu、Siyuan Huang
- **机构：** 清华大学、北京通用人工智能研究院（通用人工智能国家重点实验室）、北京大学
- **核对范围：** arXiv HTML v1 的引言、相关工作、§3 方法及 Algorithm 1、§4 Tables 1–4、结论、附录 A–D 的关键推导、参数、评估协议及局限；尚未逐行核完全文及图像细节。
- **通读：** 未完成
- **入库日期：** 2026-11-24

## 核心摘录与理解

1. §3.1/Algorithm 1：从带位姿 RGB 图像及实例掩码建立 GeoSVR 稀疏八叉树体素；按对象和背景拆分，几何不确定度用于构造锚点。背景通过 See3D 补色、平面深度约束重新优化并提取网格。
2. §3.2/Appendix A：冻结 TRELLIS，在原场景坐标内优化结构与外观潜变量。预测噪声残差由速度场换算，按 SDS 忽略模型 Jacobian 的推导，每个采样梯度相较速度目标多一个 (1-t) 因子，抑制高噪声不可靠更新。不能把这个逐采样权重误读成任意完整梯度的恒等式。
3. §3.3/Appendix B：occupied anchor 保存高置信结构；vacant anchor 由至少三视图的可见自由空间及其他实例/背景确定。空区潜变量向 void 分布重采样，而非硬置零；锚权重、噪声范围按迭代退火。
4. §3.4：先优化几何占据，再将 DINOv2 多视图特征反投影初始化外观潜变量，解码得到每对象带纹理网格。
5. Table 1：Replica / ScanNet++ 的 CD 分别为 3.95 / 4.34 cm；F-score（5 cm）77.87 / 77.93，PSNR 24.21 / 21.71 dB。Table 4 Replica 耗时 1.35 h，峰值显存 20.965 GB；均为作者实验，不是本次复现。
6. Appendix D：对初始重建、掩码及先验分布敏感；主评测为室内可实例分割环境；逐物体优化计算代价较高。输出是视觉网格而非带物性仿真资产。

## 对 wiki 的映射

- [DecomVoxel 独立论文实体](../../wiki/entities/paper-decomvoxel.md)
- [官方项目页](../sites/decomvoxel.md)
- [官方源码](../repos/decomvoxel.md)
