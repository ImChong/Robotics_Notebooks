# Transformer-Based Trajectory Prescription for sEMG-Optimizing Robot-Mediated Rehabilitation（arXiv:2611.08295）

> 来源归档（paper）

- **标题：** Transformer-Based Trajectory Prescription for sEMG-Optimizing Robot-Mediated Rehabilitation
- **类型：** paper / rehabilitation-robotics / sEMG / transformer / flow-matching
- **arXiv：** <https://arxiv.org/abs/2611.08295>
- **PDF：** <https://arxiv.org/pdf/2611.08295v1>（仅 v1，首页标注 2026-11-05，cs.RO）
- **作者：** Gabriel Parra、Kimia Khoshnami、Daniel S. Brown、Laura A. Hallock
- **机构（论文首页）：** 犹他大学（University of Utah）机械工程系；Kahlert 计算学院（Kahlert School of Computing）
- **发表信息：** 论文页脚写明 10th Conference on Robot Learning（CoRL 2026），Austin TX
- **项目页 / 数据：** SimTK 项目 OpenRobotRehab：<https://simtk.org/projects/openrobotrehab>（论文摘要称数据与代码在此开源）
- **入库日期：** 2026-11-09
- **一句话说明：** 用 RoPE Transformer 自编码器把 8 通道上肢 sEMG 时间序列映射为 x/y/z 目标轨迹（经力—速度映射，等长任务），再用流匹配在潜变量上给出分布与不确定性；16 名健康参与者，无临床人群数据。

## 通读记录

- **核对范围：** 逐段读完 arXiv v1 PDF 正文（摘要、1 引言、2 相关工作、3 系统设计含 3.1–3.3、4 实验含 4.1 基线与 4.2 微调、局限、结论、参考文献）以及附录 A–I（A 参与者信息、B 协议偏差、C 锁轴/不锁轴任务、D 任务轨迹图、E sEMG 处理、F 超参数、G 基线与消融、H 各用户闭环图、I 微调贪心算法）；表 1–4、图 3–5 对照原图核对；各轨迹示意图（图 7–14）、图 15–18 只看了说明文字与趋势，未逐个核对曲线。参考文献只读到条目层面，未读被引论文。SimTK 项目页只读到公开首页文字，数据 / 代码下载需另行登录或点击下载页，未下载、未运行任何代码。
- **通读：** 已完成

## 核心摘录

1. **任务定义（§3.3）：** 目标是学习 g≈f⁻¹，使 v̂_target = g(a)；a 为 8 通道 sEMG（T 步 × 8），v_target 为 T 步 × 3 的 x/y/z 速度轨迹；引入未知潜变量 U 表示个体神经肌肉策略差异，a = f(v_target, U)。
2. **平台（§3.1，图 1）：** KUKA LBR iiwa 14 R820 末端执行器 + Bota SensONE 六轴力传感器；用户施力映射为屏幕虚拟球的 x/y/z 速度（z 用球大小表示）；力 1000 Hz，游戏轨迹 60 Hz，sEMG 2.22 kHz，8 块上肢肌肉（前 / 中 / 后三角肌、肱二头肌、肱三头肌长头、肱桡肌、腕屈肌、腕伸肌）。
3. **数据（§3.2，表 1，附录 A、B）：** 16 名健康参与者（P01–P16；5 女 11 男；年龄 28.4 ± 7.0，19–47；5 左利手 11 右利手），全部用右臂；每人 15 个等长任务（T01–T12 为单轴，T14–T15 为平滑布朗运动随机轨迹；表中无 T13）。P01 的力水平与后来参与者不同（附录 B）。
4. **模型（§3.3，图 2，式 1，表 4）：** RoPE Transformer 编码器得 z∈R⁶⁴，拆成 z_v（32 维，对应目标轨迹）与 z_U（32 维，对应个体行为）；轨迹解码器从 z_v 重构 v̂_target，活动解码器从 z_v、z_U 重构 â（仅训练用）；流匹配模型学习 p(z_v | z_U)，可采样得到轨迹分布；总损失为三项加权 MSE。表 4：宽度 128、4 头、前馈 512、3 层编码 / 3 层解码、流匹配隐层 256、ODE 20 步。
5. **训练与划分（§4.1）：** 训练用 P01–P10 的 T01–T12；测试分三种：已知用户新任务（P01–P10, T14–T15）、新用户已知任务（P11–P16, T01–T12）、新用户新任务（P11–P16, T14–T15）；6480 个训练窗口、4562 个测试窗口；10 s 窗口、1 s 步长；AdamW、学习率 1e-3、batch 64、100 轮；RTX 4070 上基线训练约 1.5 h；单条轨迹推理约 5 ms，250 个流匹配样本 <1 s。
6. **主结果（§4.1，表 2）：** Full 模型流匹配均值预测 MSE_FM：新任务 0.067 ± 0.190、新用户 0.077 ± 0.211、新用户新任务 0.062 ± 0.174；确定性解码 MSE_TD 约 0.084 / 0.086 / 0.082；文中称确定性误差约 8.5%，流匹配带来约 1–2% 提升。标准差远大于均值（文中亦指出），文中归因于分布外区间被流匹配拉回训练分布。LSTM 在新任务（0.068 ± 0.143）与新用户新任务（0.063 ± 0.128）上的确定性误差与 Full 的 MSE_FM 相当，但新用户已知任务上为 0.191 ± 0.354，训练集上为 0.155 ± 0.308。MSE 的归一化口径论文未给出公式之外的说明。
7. **闭环初测（§4.1「Preliminary…」，图 4、附录 H）：** 取 P07 的 T08 目标活动，用 Full 模型（250 样本期望）生成一条轨迹；5 名参与者（P08、P13，加新增的 P17–P19）跟随，只报告最后一次重复；活动 RMSE 均值 0.155、标准差 0.026。
8. **微调（§4.2，图 5，附录 I）：** 对 6 名测试用户（P11–P16），用贪心算法逐个选 1–12 个用户任务再训练；所有用户损失 L（式 1）随数据增加下降，效应大小约 2–12%；12 个模型共约 7 min，每个约 35 s；校准总时间 = 新增试次数 ×（60 或 20 s）+ 35 s。
9. **论文自述局限（Limitations 节）：** 只有健康人、等长任务；任务未优化；流匹配在分布边缘的行为只做了定性检查，不构成安全保证；超参数凭经验；未考虑利手等因素；后续要做病理人群、动态轨迹。

## 开源状态（步骤 2.5）

- 论文摘要与结论称数据与代码作为 OpenRobotRehab 项目发布在 SimTK。
- SimTK 项目首页（2026-11-09 访问，页面显示最后更新 2026-07-28）可打开，首页文字描述的是 OpenRobotRehab 1.0 发布：13 名健康、2 名卒中参与者的 sEMG / 力 / 游戏数据与分析代码（对应 ICORR 2025 平台论文）。**未核实**：该页面当前是否已挂出本文 16 名健康参与者的数据与本文模型训练代码；论文所称开源内容的具体文件未下载核对。

## 对 wiki 的映射

- [paper-transformer-trajectory-prescription-semg-rehab](../../wiki/entities/paper-transformer-trajectory-prescription-semg-rehab.md)
