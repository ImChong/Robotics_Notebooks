# Splat2Act: From Captured Scenes to Personalized VLA Manipulation（arXiv:2610.13945）

> 来源归档（ingest）

- **标题：** Splat2Act: From Captured Scenes to Personalized VLA Manipulation
- **短名：** Splat2Act
- **类型：** paper
- **作者：** Xiaomeng Ye、Krishna Jaganathan、Gyubeum Lim、Gunjan Chhablani、Muhammad Zubair Irshad、Zsolt Kira
- **机构：** Georgia Institute of Technology；Waymo；Toyota Research Institute
- **arXiv：** <https://arxiv.org/abs/2610.13945>
- **PDF：** <https://arxiv.org/pdf/2610.13945>
- **项目页：** <https://splat2act.github.io/>
- **入库日期：** 2026-10-13
- **一句话说明：** 仅需手持场景扫描与物品照片，构建楼宇级 3DGS 数字孪生，由仿真 oracle 自动生成拾取演示，并两阶段微调 π₀.₅ 到目标场景。

## 核心摘录（面向 wiki 编译）

1. **真实到仿真：** 手机 RGB-D 视频重建完整室内空间；3DGS 保存外观，提取网格支持碰撞与任务规划。系统发现承载面并采样可达拾取任务。
2. **仿真数据闭环：** 几何 oracle 规划并执行拾取，失败时重试；人的输入止于场景和物品采集，训练演示来自仿真。
3. **两阶段策略适配：** 先用公开物品示范适配预训练 π₀.₅，再用用户自己的物品和部署场景承载面个性化。
4. **规模与指标：** 101 个校园扫描场景、中位可导航面积 104.87 m²、852 个承载面、68,021 条成功拾取演示；Office 目标场景成功率 62.5%。同一 12 项真机子集 Stretch 3 为 63.9%（36 次尝试），仿真为 66.7%（12 episodes）。
5. **边界：** 未适配 π₀.₅ 在目标场景拾取成功率为 0%；未见承载面上的成功率下降。论文验证 receptacle-local pickup，较长时程移动操作留待后续。

**对 wiki 的映射：** [paper-splat2act-personalized-vla](../../wiki/entities/paper-splat2act-personalized-vla.md)

## 版本与引用

- arXiv v1，提交于 2026-10-09。
- 以论文与项目页公开数据为准；各百分比任务数量与分母不同，阅读时需区分。

## 推荐核查点

- 真机 63.9% 的 12-task 子集与 36 次尝试协议。
- 目标场景个性化与跨场景泛化的差异。
- 官方 GitHub 和数据集发布后，补充源码运行路径、许可与复现步骤。
