# Grasping the Future（arXiv:2611.12879）

> 来源归档（paper）

- **标题：** Grasping the Future: Joint 3D Hand–Object Forecasting From Internet Videos
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.12879>
- **PDF：** <https://arxiv.org/pdf/2611.12879v1>（v1，2026-11-09，cs.CV）
- **项目页：** <https://grasping-the-future.github.io/>
- **代码：** 无。论文 §4.1 写「We will release the code and data publicly」；项目页（2026-11-11 访问）Code / Model / Data 三个入口均标「soon」，未列 GitHub 链接。
- **作者与机构：** Rustin Soraki、Matthew Wallingford、Ryan Aby、Brandon Chee（华盛顿大学）、Jason Ren（北卡教堂山）、Ali Farhadi（UW）、Homanga Bharadhwaj（约翰斯·霍普金斯）、Roozbeh Mottaghi（Skild AI）；后三位为 equal supervision
- **入库日期：** 2026-11-11
- **一句话说明：** 用掩码时空补全 + 流匹配，一个模型统一做手—物体三维未来预测、跨实体预测、目标条件生成与时间补全；用 6.2 万条自互联网视频自动重建的三维轨迹作监督；预测经 SPIDER 重定向到仿真灵巧手验证物理可行性。

## 核对范围与通读

- **核对范围：** 论文 v1 PDF 正文 §1–§5 与参考文献、补充材料 Appendix A–F（模型与训练细节、提取管线细节与提取质量、基准细节、消融、能量分数与任务专家对比、局限）的文字与表格全部逐段读过；图（Fig. 1–6）只读了图注与图内文字标注，未逐图检视像素级细节；项目页静态读了页面文字（未播放视频，未下载 paper.pdf / watch3d.html）。未读被引文献。
- **通读：** 已完成

## 核心摘录

1. **任务统一（§3.1，Table 3）：** 轨迹按 5 s 窗口；每帧 5 个 token（左右手、物体、两个手—物体关系 token）；用 mask 区分 5 类任务（预测、手→物、物→手、目标条件、时间补全），评测拆成 7 列任务。
2. **模型（§3.2，Appendix A）：** 12 块 Transformer、宽 512、8 头，103.3M 可训练 + 12.7M 冻结参数；推理 10 步 Euler；训练 40,000 步，16 张 H100 约两天（Table 4）。
3. **损失（§3.3）：** 流匹配 L_FM + 手—物体附着损失（接触帧，指尖距物体表面 ≤5 mm）+ 对 M=3 条样本的能量分数损失。
4. **数据（§3.4，§4.1）：** 候选 clip 来自 Action100M；约 12% 通过验证；共 74,312 条轨迹 / 134,661 个窗口，其中互联网视频 62,562 条 / 109,346 窗口；8 个动捕/标注数据集。Fig. 3：36% 未通过动作验证、52% 在提取或质检阶段被拒、12% 保留。
5. **基准（§4.1，Appendix C）：** 331 个五秒窗口（256 来自标注数据集、75 来自互联网视频，互联网视频的参照为论文自己提取的轨迹，非真值）；每个输入采样 20 条。
6. **结果（§4.2，Table 1/2）：** 与已发布预测器对比，能量分数下 ADE/FDE 低 16–37%（Table 2）；单样本预测时 stay-put 对未来预测反而更好（高 2–8% 误差）；MotionForesight 单样本物体 ADE/FDE 比本文单样本低但几乎不动物体。
7. **重定向（§4.3，Fig. 5）：** 103 条可被参照轨迹成功重定向的样例，每例 3 个 SPIDER 随机种子；运动学重定向 14.6%，加抓取预备 43.7%，SPIDER 动态优化 73.8%；全部在仿真（XHand）中。
8. **局限（Appendix F）：** 预训练感知模型的错误进入训练数据（深度尺度歧义、WiLoR 指尖丢失、SAM 3D 朝向翻转、细长物体失败、MegaSaM 在三脚架镜头下的伪运动）；候选 clip 大量被拒、提取算力高（Appendix B.3：约 2.8 GPU 分钟/候选 clip，约 28,700 H100 小时）。

## 对 wiki 的映射

- [paper-grasping-the-future](../../wiki/entities/paper-grasping-the-future.md)
