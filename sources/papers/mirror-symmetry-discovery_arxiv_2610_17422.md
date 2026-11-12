# MIRROR（arXiv:2610.17422）

> 来源归档（paper）

- **标题：** MIRROR: Reflecting Hidden Symmetries from Offline Robot Interaction Data
- **作者：** Shiqi Liu, Yihang Yao, Peide Huang, Mengdi Xu, Ding Zhao（Carnegie Mellon University；Mengdi Xu 署名 Stanford University）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.17422>（v1，cs.RO，2026-10-12）
- **PDF：** <https://arxiv.org/pdf/2610.17422v1>
- **项目页：** <https://pomdp-mirror.github.io>（论文摘要给出；未见代码仓库链接）
- **入库日期：** 2026-11-12
- **一句话说明：** 在已知对称群 G 的前提下，用离线单步转移元组 (o, a, o', r) 的对抗匹配加群公理正则，学出观测与动作耦合的非线性群作用，并用于帧平均行为克隆。
- **核对范围：** 通读 arXiv v1 PDF 正文全部 9 页（摘要、第 1–8 节含限制节）与附录 A（群作用公理与引理）、B（仿真设置、奖励、数据收集、评估指标）、C（网络结构与超参数）、D.1 的 Tables 6–9 与 D.3.1 文字、E（帧平均策略与 Table 10 的文字与 C16 行）、F（硬件、预处理、仿真 Table 11、真机 Table 12）；Table 10 的 SymGAN / LaLiGAN 其余子群行、D.2 的热图与逐任务曲线、附录图像本身只看了题注与文字描述，未逐图细看；未运行任何代码。
- **通读：** 已完成

## 开源状态

- 论文给出项目页 pomdp-mirror.github.io；正文与附录均未给出代码仓库地址（2026-11-12 核查论文文本），故 wiki 页不写 `code:`。

## 核心摘录

1. **问题（第 1 节）：** 对称感知策略需要观测与动作耦合变换；像素观测下，旋转抽屉既改变外观又改变拉开方向，对应的图像变换依赖场景几何与视角，一般不是图像旋转。现有方法靠点云或预设的已知群作用表示（第 2 节）。
2. **方法（第 4 节）：** 两阶段。阶段 1 预训练并冻结感知自编码器 (E_o, D_o)；阶段 2 用观测对称模块 (E_z, D_z) 与动作对称模块 (E_a, D_a)，在固定线性群表示 ρ_z(g)、ρ_a(g) 下编码、变换、解码；判别器对潜变量转移元组 (z, a, z', r) 做对抗匹配（式 1），奖励不变；三个正则 L_id、L_comp、L_inv（式 2）加非平凡惩罚 L_nt（式 3，沿用 LaLiGAN）。
3. **实验设置（第 5 节、附录 B）：** G = SO(2)，4 个自定义 Robosuite 任务（Drawer Opening / Drawer Closing / Cube Dropping / Hammer Lifting），固定相机 RGB，每任务 10 万条训练元组与 1 万条评估元组，由有限状态机控制器采集（B.4），3 个种子。
4. **对称质量（第 5.2 节，Table 1）：** 汇总对称不变误差 0.157、相容误差 0.717、可逆误差 0.730（均最低）；恒等误差 0.317 高于 LaLiGAN 的 0.296。相容误差相对 LaLiGAN 的 1.838 降低 61%。去掉 L_adv 后不变误差升至 1.701。
5. **策略学习（第 5.3 节，Table 2、附录 E.4 Table 10）：** 帧平均 BC（EBC）在 H = C16 下 EBC-MIRROR 平均成功率 85.1%，普通 BC 80.5%（正文摘要写作 85.12% 对 80.48%）；Cube Dropping +9.66 个百分点最大。手工设计的 ExSym 与 GAN 发现的 SymGAN、LaLiGAN 帧平均变体平均成功率均低于普通 BC。
6. **真机（第 6 节、附录 F）：** 换成深度导出的高度图加 SAM 2 目标分割掩码两通道观测，在仿真中重新发现对称并训练，零样本部署到 Kinova GEN3，每任务 20 次：13/20、19/20、9/20、16/20（Table 12）。真机未与普通 BC 对比。
7. **限制（第 8 节）：** 需事先指定群 G；只能恢复数据中已存在的对称性，数据缺乏沿对称方向的变化时变换可能塌缩为恒等。

## 对 wiki 的映射

- [paper-mirror-symmetry-discovery](../../wiki/entities/paper-mirror-symmetry-discovery.md)
