# Scalable Behavior Cloning with Open Data, Training, and Evaluation（ABC，arXiv:2606.27375）

> 来源归档（paper）

- **标题：** Scalable Behavior Cloning with Open Data, Training, and Evaluation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.27375>
- **PDF：** <https://arxiv.org/pdf/2606.27375v1>
- **项目页：** <https://abc.bot>（论文首页给出；代码仓库地址论文未给出，正文称完整栈「将发布」）
- **作者与机构：** Allshire、H. G. Singh、R. Singh、Rashid、Choi、McAllister 等（共同一作，Amazon FAR 实习期间完成），UC Berkeley / MIT / Amazon FAR / XDOF / CMU；资深作者含 Abbeel、Isola、Malik、Kanazawa（arXiv v1，2026-06-25）
- **入库日期：** 2026-11-12
- **一句话说明：** 面向双臂低成本平台（I2RT YAM）的开源行为克隆全栈：ABC-130K 真机遥操数据（3,553 小时、134,806 条、195 任务）、DiT 与 VLA 两类策略的架构与算力消融、MuJoCo 仿真（400 小时仿真遥操）及仿真与真机评测相关性。
- **核对范围：** 用 pdftotext 通读 arXiv v1 PDF 全部 30 页：正文第 1 至 8 节、参考文献、附录 A 至 H（含表 1 至 5）；目视核对了第 10、11 页的图 11 至 15（柱状图数值为读图估计，未逐点量取）；其余图表只读了题注与坐标文字，未逐图细看。未下载或运行任何代码与数据；项目页 abc.bot、数据集与权重发布页未访问，因此「已开源」状态以论文自述为准、未独立核实。
- **通读：** 已完成

## 开源状态

- 论文自述（图 2 题注）：完整栈「will be released」，即发布时点在论文写作时仍为将来时；摘要与引言写「open-source」。本归档未核实实际发布情况。

## 核心摘录

1. **数据（第 2 节）：** ABC-130K 为 134,806 条 episode、195 任务、3,553 小时双臂遥操数据；任务分 7 类（附录 A：抓放 67 任务 793 h，精细抓放 39 任务 736 h，折叠 36 任务 883 h，插入 19 任务 441 h，工具使用 16 任务 321 h，分类 8 任务 205 h，系带 10 任务 175 h）；1,552 小时子集含子目标标注。
2. **硬件（第 6.1 节、附录 C.1）：** 两台 I2RT YAM 6 自由度机械臂，三台 RealSense D405（一顶视两腕部），30 Hz；数据在三面白墙围栏内采集；论文称整机约 8,000 美元。
3. **模型（第 3 节）：** ABC-DiT（DINOv3 ViT-B 加池化交叉注意力，32 层、隐层 1536、24 头，总计约 2B 参数）与 ABC-VLA（Gemma 3 4B 加小型 DiT 动作头，池化 adaLN 连接，8 次扩散噪声采样的方差缩减）；输入三路 224x224 图像加 14 维关节状态，输出 30 步绝对关节位置块。
4. **架构消融（表 1）：** 在内部 7,000 小时语料上训练 200k 步、三任务各 50 次真机试验：DINOv3-xattn 平均进度 67.5%，Pooled adaLN VLA 61.4%，CLIP-xattn 58.8%，CLIP-adaLN 47.3%，FAST+X-Attn 32.6%，X-Attn VLA 11.7%。
5. **批量与离线指标（第 3.3、3.4 节）：** 批量 1536 / 4608 / 9216；小批量时 DiT 优于 VLA，9216 时 VLA 提升更大（图 5）；训练损失与验证动作误差和真机表现显著负相关（图 8，n=16，验证动作误差 Pearson r=-0.89 严格成功率，-0.85 进度），验证损失不显著相关。
6. **仿真相关性（第 4 节、图 9）：** MuJoCo 中复刻真机，发布 10 个仿真任务与 400 余小时 VR 遥操数据；12 个检查点、三任务各 50 次试验，仿真与真机严格成功率 Pearson r=0.85（p=4.2e-4），任务进度 r=0.91（p=5.0e-5）。
7. **下游微调与 DAgger（第 5.2、5.3 节）：** 四个精细任务上预训练初始化优于从头训练（图 13）；折叠纸盒经 10 小时「collapsed」SOP 数据微调后进度 24%，加 DAgger 后图 14 约升至 0.85（读图估计）。
8. **策略条件化（附录 H）：** 操作员 ID 条件化、动作前缀（RTC）长度、子任务条件化，均在 t-shirt 折叠上用 CLIP 交叉注意力版 ABC-DiT 评测（每条件 n=10）。

## 对 wiki 的映射

- [paper-abc-scalable-behavior-cloning](../../wiki/entities/paper-abc-scalable-behavior-cloning.md)
