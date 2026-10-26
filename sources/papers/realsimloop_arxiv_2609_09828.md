# RealSimLoop 论文来源归档

## 元数据

- **题目：** RealSimLoop: Online Real-to-Sim Adaptation via Differentiable Reduced-Order Simulation with Vision Feedback
- **作者：** Zhihao Cen, Chuhua Xian, Hailin Sun, Yuliang Liufu, Zhen Zhang, Xiangyu Chu, Hongmin Cai, Yunbo Zhang, Guoxin Fang
- **单位：** 华南理工大学（SCUT）、香港中文大学（CUHK）、香港科技大学广州校区（HKUST-GZ）、中国科学院香港创新研究院
- **预印本：** https://arxiv.org/abs/2609.09828
- **HTML：** https://arxiv.org/html/2609.09828v2
- **PDF：** https://arxiv.org/pdf/2609.09828
- **版本：** v2，2026-09-14；cs.GR，交叉分类 cs.CV、cs.RO
- **发表信息：** SIGGRAPH Asia 2026 Conference Papers，DOI: https://doi.org/10.1145/3829340.3842376
- **项目页：** arXiv 页面未列独立项目页；作者团队论文列表：https://guoxinfang.github.io/pub.html
- **代码：** 论文称录用后发布；截至 2026-10-26，arXiv 页面和作者论文列表未提供代码仓库链接

## 论文要解决的问题

离线 Real-to-Sim 通常用一段预采集序列拟合固定材料参数，遇到长时交互、材料随时间变化或模型误差时会产生漂移。纯视觉重建能跟踪表面几何，却难以恢复内应力、作用力等隐藏物理量。RealSimLoop 目标是用视觉反馈持续校准可微软体仿真，使数字模型跟随真实物体的动态变化，并支持后续物理量估计。

## 方法与关键摘录

- **滑动窗在线目标：** 对最近 w 帧的视觉观测与仿真渲染结果累计误差；每个时间窗更新随时间变化的材料参数，减少整段开环 rollout 的误差累积。
- **神经降阶动力学：** 用数据驱动的非线性子空间映射把高维网格状态压缩到低维潜变量 z，在该子空间内求解可微物理仿真。论文采用 AutoEncoder 子空间，编码器和解码器均为两层、每层 200 个隐藏单元。
- **视觉到物理的梯度链：** 将软体几何映射到 3D Gaussian Splatting（3DGS），通过可微渲染计算像素损失，再把梯度传回降阶状态和材料参数。材料梯度使用伴随法沿时间窗反向传播，避免显式构造整条仿真轨迹的稠密雅可比。
- **训练与实现：** 降阶可微模拟模块使用 JAX，3DGS 使用 PyTorch，真实视觉数据用 SAM 2 处理；训练和实验使用 NVIDIA RTX 4090。训练子空间的数据由不同材料参数和随机交互序列组合生成。
- **对照结果：** 双 UR5e 操作硅胶软杆、4 个相机时，平均重建误差为 4.6 mm（约模型尺寸的 2.3%）；降阶方案将每帧在线更新耗时从全空间方法的 3.30 s 降至 0.53 s，约 6.23× 加速。论文表中不同场景加速范围为 2.52×–7.69×。
- **在线性的收益：** 对材料老化等时变刚度场景，在线方法能跟踪杨氏模量变化；相较离线方案，逐帧误差降低约 3–5×。视觉反馈相较稀疏 marker 输入，在恐龙模型测试中把顶点 L2 误差从 0.052 m 降至 0.026 m。
- **下游用途：** 适配后的模型用于双臂形状控制，并由重建应力场估计外部拉索力；论文报告估计力与传感器测量相符。还展示应力场重建与新视角可视化。

## 解释边界

- 论文中的“quasi-real-time”指其具体测试设置；表中更新耗时随模型、子空间维数、时间窗和相机数量变化，不能直接当作任意系统都能达到的控制周期。
- 高速弹球场景中降阶方法未收敛，该场景只报告全空间结果；降阶表示并非对所有高动态形变都有效。
- 更大的子空间维数和时间窗可提高精度，但增加更新时间；相机视角的收益也会在覆盖主要几何后趋于饱和。
- 论文称源码将在录用后发布，但截至归档日未提供公开代码链接，当前无法据此复现实验。

## 对 wiki 的映射

- **论文实体页：** [RealSimLoop](../../wiki/entities/paper-realsimloop.md)
- **关联方向：** [Real2Sim / Sim2Real](../../wiki/concepts/sim2real.md)、[CRISP Real2Sim](../../wiki/methods/crisp-real2sim.md)、[One-Shot Real-to-Sim](../../wiki/entities/paper-sa-2412-00259-one-shot-real-to-sim-via-end-to-end-differentiab.md)

## 一手来源

- arXiv HTML（论文全文）：<https://arxiv.org/html/2609.09828v2>
- arXiv 摘要与版本记录：<https://arxiv.org/abs/2609.09828>
- 作者团队论文列表：<https://guoxinfang.github.io/pub.html>
