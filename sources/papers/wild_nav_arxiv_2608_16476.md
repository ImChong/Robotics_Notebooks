# Exposing the Long-tail in Embodied Urban Navigation via Scalable Learning from In-the-Wild Videos（arXiv:2608.16476）

> 来源归档（ingest）

- **标题：** Exposing the Long-tail in Embodied Urban Navigation via Scalable Learning from In-the-Wild Videos
- **类型：** paper / embodied-navigation / VLA / egocentric-video / long-tail
- **arXiv abs：** <https://arxiv.org/abs/2608.16476>
- **PDF：** <https://arxiv.org/pdf/2608.16476>
- **作者：** Bingyi Xia、Han Bao、Zhewei Chen、Hanjing Ye、Jingwen Yu、Yuhan Pang、Wenjun Xu、Jiankun Wang
- **机构：** 鹏城实验室（Peng Cheng Laboratory）、南方科技大学（Southern University of Science and Technology）、香港科技大学（The Hong Kong University of Science and Technology）
- **发表日期：** 2026-08-17（arXiv v1）
- **模型 / 基准：** WILD-Nav / WILD-LongTail
- **项目页与开源：** 论文及作者发表页未提供官方代码仓或数据集下载入口（核查日期：2026-10-12）；见[作者发表页核查](../sites/wild-nav-author-publications.md)
- **入库日期：** 2026-10-12
- **一句话说明：** 把约 500 小时、跨 30 个城市的网络第一人称步行视频转成带米制轨迹和结构化导航推理的监督数据，训练 WILD-Nav 点目标城市导航 VLA，同时用“模式稀有度 + 模型误差 + 反思诊断”挖掘长尾失败并构建 WILD-LongTail。

## 核心摘录（面向 wiki 编译）

### 1) 数据从哪里来、如何变成导航样本

- 作者整理约 **500 小时**街道步行第一人称视频，覆盖 **30 个城市、15 个国家**，从中生成 **50 万以上**导航样本。
- 视频统一切成 5 分钟片段；使用 LoGeR 长上下文几何重建与 Pi3X 估计相机位姿、米制深度并形成地面轨迹。
- 以 1 Hz 采样，再切成 10 帧 episode：5 帧历史 + 5 帧未来；按时间块切 train/test，避免高度重叠的窗口泄漏到不同 split。
- 规则根据速度、侧向位移和方向把轨迹映射到 **20 种 meta-action**。GroundingDINO 提供相关物体的类别、图像位置与深度；Qwen3.6VL-27B 生成 perception–analysis–planning 结构化推理。
- 教师模型在训练标注阶段可以查看未来帧和规则动作作特权校验；论文说明这些未来信息只用于检查/纠错，要求最终推理依据可观测的历史与当前帧。

### 2) WILD-Nav 策略接口

| 项目 | 论文设定 |
|------|----------|
| 任务 | 无预建地图的局部 point-goal 城市导航 |
| 视觉输入 | 单目 RGB 当前帧 + 过去 4 帧（共 5 帧） |
| 状态输入 | 短程自我运动轨迹历史和局部 point goal |
| 输出 | 5 个未来地面轨迹 waypoint、20 类 meta-action、结构化导航 rationale |
| VLA backbone | Qwen3.5VL-4B；视觉编码器训练时冻结 |
| Action expert | 由 planning token 的隐藏状态解码 meta-action 分类和连续轨迹；轨迹头是 MLP |
| 训练目标 | rationale 自回归语言损失 + meta-action 交叉熵 + waypoint MSE |

WILD-Nav-think 先生成结构化 rationale，再据 planning token 输出动作与轨迹；WILD-Nav-inst 采用更直接的指令条件输出，作为效率对照。

### 3) 如何定义和诊断长尾

论文把每条导航 episode 表示为**视觉 / 目标 / 历史运动模式及其规划结果**，沿两个不同维度找尾部：

- **稀有（rare）：** 感知–运动联合模式在数据分布中出现较少，依据归一化特征的平滑 IDF 分数评估。
- **困难（hard）：** 模型对该 episode 的轨迹规划误差高，以 ADE / FDE 等 model-dependent error 衡量。
- **反思归因：** 教师模型对困难样本进行特权反思，将反复出现的错误映射到感知、预测、规则、风险、动作或轨迹形状等环节。
- **基准固定：** WILD-LongTail 固定挖掘出的稀有 / 困难 episode，用于跨模型一致评估；不能把稀有与困难当成同一集合。

### 4) 结果与重要限制

- 视频测试集上，WILD-Nav-think 的 ADE / FDE / MAOE 为 **0.086 m / 0.068 m / 5.32°**；WILD-Nav-inst 为 **0.094 m / 0.071 m / 5.90°**。
- 相比 SocialNav，WILD-Nav-inst 在视频集三个规划误差上至少降低 53.4%；think 变体比 inst 的 ADE 和 FDE 再降低 8.5% 与 4.2%。
- 直接迁移到真实机器人数据时存在明显视角和运动约束 domain gap；真实数据微调能提升真实数据测试表现，但不完全消除差距。
- 长尾微调中，仅加 hard 样本可改善固定 hard tail，却损害整体误差；hard + random replay 在整体能力与 tail 指标之间取得更好的平衡。
- 论文展示移动机器人在行人交互、路口转弯、人行道障碍和信号灯路口的定性轨迹结果；这些结果不等于公开了完整实时部署系统。

## 项目开放状态核查

- 截至 **2026-10-12**，arXiv 论文没有列出 GitHub、项目网站、模型权重或数据集下载链接；作者的公开发表列表只链接到论文。
- 论文描述的数据规模和 WILD-LongTail 基准不代表数据集已公开可下载。当前可核实的是论文及结果描述；官方训练代码、策略权重、原始视频和基准文件的公开状态**未确认 / 未链接**。
- 若后续发现作者公开 release，应更新同一实体及本归档，不另建 repo 占位节点。

## 对 wiki 的映射

- 新建统一项目实体：[WILD-Nav / WILD-LongTail](../../wiki/entities/paper-wild-nav-longtail.md)。
- 交叉：[Vision-Language Navigation](../../wiki/tasks/vision-language-navigation.md)、[VLA](../../wiki/methods/vla.md)。
