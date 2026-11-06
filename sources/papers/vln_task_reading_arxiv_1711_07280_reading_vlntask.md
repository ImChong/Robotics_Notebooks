# R2R / Matterport3D Simulator 论文通读（VLN 任务奠基视角）

> 来源归档（原文通读 · 支撑 `wiki/tasks/vision-language-navigation.md`）

- **标题：** Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments
- **作者：** Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, Anton van den Hengel（ANU / Adelaide / QUT / Macquarie）
- **类型：** paper / vln / benchmark（CVPR 2018）
- **原文：** <https://arxiv.org/abs/1711.07280>（2026-11 通过 arXiv PDF v3 获取全文）
- **一句话说明：** 提出 Matterport3D Simulator 与 Room-to-Room（R2R）数据集，首次把「真实未见建筑 + 自然语言指令 → 动作序列」定义为可复现评测的 VLN 任务，并给出 seq2seq 基线。
- **核对范围：** 通读 arXiv v3 PDF 正文 §1–§7（引言、相关工作、§3 模拟器、§4 R2R 任务与数据集、§5 agent 与基线、§6 结果、§7 结论）及图表说明；参考文献仅浏览；表 1 逐格数字以正文引用为准，未另行复现实验；补充材料（标注界面）未获取。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **任务动机（§1）：** VLN 与 VQA 同属「视觉接地的序列到序列转换」，但 VLN 序列更长，且输出的是改变相机视角的动作序列；强调指令须在**未见过的真实环境**中解释，才是开放集问题（渲染图或标签化环境会退化为闭集分类）。
- **模拟器（§3）：** 基于 Matterport3D（90 个建筑尺度场景、10,800 个全景视点、194,400 张 RGB-D 图）；智能体位姿 = 视点 v、朝向 ψ∈[0,2π)、俯仰 θ∈[−π/2,π/2]；观测为由预计算立方体贴图透视投影得到的第一人称 RGB 图。
- **导航图与动作空间（§3.2.2，式 1）：** 每个场景带加权无向图 G=⟨V,E⟩，边由网格射线检测得到，剔除 >5 m 的边并人工核对（补窗、镜面）；下一步可达视点 = 当前视点 ∪ 图中相邻且处于视野内的视点；平均每图 117 个视点、平均度 4.1；动作确定性；运动被离散到视点（作者承认这是限制，也论证对高层任务影响不大）。
- **偏差（§3.2.4）：** Matterport 场景偏整洁豪华、几乎没有人和动物、视点偏「制高点」；这些是数据的选择 / 采集偏差。
- **R2R 构造（§4）：** 起终点多在不同房间，取导航图最短路径，丢弃 <5 m 以及边数 <4 或 >6 的路径，共 7,189 条，平均 10 m；每条路径由 AMT 工人写 3 条指令，共 21,567 条，平均 29 词，词表约 3.1k（约 1.2k 出现 ≥5 次）；400+ 工人、约 1,600 小时；指令抽象程度差异大。
- **评测（§4.4）：** 导航误差 = 终点与目标在导航图上的最短路径距离；< 3 m 记成功；不评整条轨迹（指令常不指定路径）；必须主动 stop；另报 oracle 停止规则成功率。划分：train 14,025 / val seen 1,020 / val unseen 2,349（11 场景）/ test 4,173（18 场景，目标不公开，走评测服务器）。
- **基线模型（§5.1）：** LSTM seq2seq + 注意力；指令词序反转；图像用 ImageNet 预训练 ResNet-152 均值池化特征，与上一步动作嵌入拼接；动作简化为 6 个（left/right/up/down/forward/stop）；训练对比 teacher-forcing 与 student-forcing（等价于在线 DAGGER，实验中优于 scheduled sampling）；512 隐单元、Adam、batch 100、测试用贪心解码。
- **结果（§6）：** RANDOM 测试成功率 13.2%；人类（AMT）86.4%；student-forcing 在 val seen 38.6%、val unseen 21.8%，测试集榜单首个结果 20.4%；训练曲线显示 unseen 环境上很快平台化而 seen 继续提升，说明学到的视觉接地偏训练环境。
- **结论（§7）：** 现有视觉–语言方法可用于 VLN；泛化到未见环境是主要挑战；众包重建的真实场景是可扩展资源。后续方向含指令生成、具身问答、人机对话与向真实机器人迁移。

## 对 wiki 的映射

- [`wiki/tasks/vision-language-navigation.md`](../../wiki/tasks/vision-language-navigation.md)
- [`wiki/entities/paper-vln-01-r2r.md`](../../wiki/entities/paper-vln-01-r2r.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1711.07280>
- 同论文的策展摘录（非全文）：[vln_survey_01_r2r.md](vln_survey_01_r2r.md)
