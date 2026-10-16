# TacBPM: A Tactile-conditioned Behavior Prior Model for Dexterous Reorientation（arXiv:2609.18174）

> 来源归档（ingest）

- **标题：** TacBPM: A Tactile-conditioned Behavior Prior Model for Dexterous Reorientation
- **简称：** TacBPM
- **类型：** paper / dexterous-manipulation / in-hand-reorientation / tactile / behavior-prior / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2609.18174>
- **PDF：** <https://arxiv.org/pdf/2609.18174>
- **v1 日期：** 2026-09-16（UTC；仅 v1）
- **作者（arXiv 摘要页顺序）：** Jie Yin、Wanli Xing、Zeyuan Zhao、Xuezhou Zhu、Zhijie Deng、Kaifeng Zhang（通讯作者 Jie Yin）
- **机构：** Sharpa Robotics（文章转述“作者全部来自 Sharpa”；arXiv v1 HTML 署名同为 Sharpa Robotics，未列分作者机构）
- **发表信息：** 未见录用信息，按 arXiv 2026 记
- **项目页：** <https://tacbpm.github.io/>（匿名投稿样式：作者栏写 “Anonymous Authors”，Paper 与 Code 按钮均标 “(Soon)”，页脚 “TBD”）
- **代码：** **未找到**（arXiv 摘要页未列代码或仓库；项目页 Code 为 “(Soon)”；未见数据集入口）
- **开源状态（2026-10-16 核查）：** **未开源 / 待发布**。
- **入库日期：** 2026-10-16
- **文章来源：** [硬件、数据到模型：Sharpa如何系统性推进灵巧操作](../blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md)
- **一句话说明：** 把 8 个不同尺寸球体的手内重定向专家蒸馏为以触觉与本体历史为条件的变分隐空间控制器；下游任务冻结该先验，只学隐空间残差动作并微调解码器，在复杂物体与臂-手任务上显著超过原始动作空间 PPO。

## 文章转述要点（来自机器之心 2026-10-16，标注“文章转述”）

- 文章转述：TacBPM 是“运动 / 交互解耦”中的“隐空间接口”，对应自下而上路线：在仿真中训练大量原子技能，再蒸馏进统一框架，也就是文中所称的 System 0（交互脑，约 100 Hz，实时触觉）。
- 文章转述：作者全部来自 Sharpa；先在 8 个不同尺寸球体上分别训练手内重定向专家，再蒸馏进以触觉为条件的隐空间控制器；下游任务冻结先验，在隐空间学习残差动作。
- 文章转述：复杂物体多任务设置中，直接用原始动作训练的 PPO 成功率仅 1%，TacBPM 达到 70%；机械臂加灵巧手的仿真抓取任务中，部分物体成功率从从零训练的不足 1% 提升到 97%。
- 文章转述：论文显示该先验能较快适配新的物体形状、新任务和新本体。
- 文章转述：张凯峰认为自上而下的方式可能不是终局；Stanford（Bohg 与 Liu 团队）的 SimToolReal、Play2Perfect 与这条自下而上路线形成呼应。

## 核心摘录

### 1) 方法定位（arXiv 摘要）

- **链接：** <https://arxiv.org/abs/2609.18174>
- **摘录要点：** TacBPM 是面向灵巧手内重定向的触觉条件行为先验：把若干球体专家策略蒸馏为单一隐空间控制器；下游策略通过残差隐动作复用冻结的先验，相比在原始关节指令上学习可减少探索；先验以触觉与本体历史为条件。评测覆盖各向异性物体上的任意位姿迁移、指定轴旋转，以及涉及抓取、抬升、搬运与到达目标位姿的臂-手 Grasp-to-AnyPose 任务（含新工具）。摘要称训练更快、原始动作 PPO 基本失败处仍得到稳定策略，并成功 sim-to-real（手内与臂-手任务）。
- **对 wiki 的映射：** [paper-tacbpm](../../wiki/entities/paper-tacbpm.md)

### 2) 项目页：三阶段与结果速览（项目页）

- **链接：** <https://tacbpm.github.io/>
- **摘录要点：** 三阶段：(1) 8 个球体重定向教师，尺度因子 0.3 到 1.0，对应约 2.4 到 8 cm 抓取直径；(2) 以关节历史、上一步目标、接触幅值、3D 接触位置与尺度线索为条件的变分编码器 + 学习先验 + 解码器，把教师蒸馏进单一控制器；(3) 下游固定先验、围绕先验均值学习残差隐码，并为新几何微调解码器。页面自报：已见尺度 95.62%；未见尺度与物体 60.38%（无触觉 40.05%）；复杂物体多任务 70.00%；block_03 沿 z 轴连续指令旋转 342.39 度；臂-手 Grasp-to-AnyPose 多种工具平均 96.73%（原始动作 PPO 49.24%）；真机部署成功但未给数值。页面同时展示了与“无触觉”“从零训练”的真机对比与失败案例（文字描述）。
- **对 wiki 的映射：** [paper-tacbpm](../../wiki/entities/paper-tacbpm.md)

### 3) 项目页：发布状态（项目页）

- **链接：** <https://tacbpm.github.io/>
- **摘录要点：** 作者栏 “Anonymous Authors”，Paper 与 Code 按钮均标 “(Soon)”，页脚 “TBD”，BibTeX 标注 manuscript in preparation，页面许可为 CC BY-SA 4.0；无 GitHub、无数据集链接。
- **对 wiki 的映射：** [paper-tacbpm](../../wiki/entities/paper-tacbpm.md)

### 4) 方法细节（arXiv 全文 HTML v1，论文自报）

- **链接：** <https://arxiv.org/html/2609.18174v1>
- **摘录要点：** 专家：8 个球体 PPO 教师，尺度因子 0.3 到 1.0（步长 0.1），标称 8 cm 球，Isaac Sim / Isaac Lab，教师观测含特权物体信息。蒸馏：在线 DAgger 风格多教师蒸馏，环境按步幅布局分给各教师，由被指派教师给每步动作标签；损失为动作 MSE + β·KL(后验‖先验) + λ·编码器均值的时间平滑，使用 8192 个环境。结构：编码器 q(z|x,g) 含任务目标，先验 p(z|x) 不含目标，解码器 π(h,z) 输出确定性 22 维手关节位置目标；16 维高斯隐变量；MLP 隐层 (512, 256, 128)；蒸馏时编码器还接收物体最长边提示，部署用的先验与解码器不含。观测共 192 维：3 帧历史，每帧含 22 个归一化关节位置、22 个上一步目标、5 个平滑接触幅值与 5 个 3D 接触位置；论文未命名触觉传感器型号，仅在硬件部分提到“calibrated tactile forces and contact points”。下游：z_task = μ_prior + Δz，a = π(h, z_task)，PPO 输出 Δz；先验与状态归一化冻结，主设置微调解码器与输出头，冻结解码器作为消融。
- **对 wiki 的映射：** [paper-tacbpm](../../wiki/entities/paper-tacbpm.md)

### 5) 实验数字（arXiv 全文 HTML v1，论文自报）

- **链接：** <https://arxiv.org/html/2609.18174v1>
- **摘录要点：** 平台为 Sharpa Wave 22-DoF 手，每行 1000 次随机化尝试（另注除外），成功为超时或掉落前达到目标朝向 0.2 rad 内。Table I（成功率 %，从零训练 / 无触觉 / TacBPM）：已见尺度均值 26.03 / 93.67 / 95.62，未见均值 11.47 / 40.05 / 60.38；sphere 1.2 为 3.30 / 1.70 / 13.30；trash_can_01 为 11.70 / 14.60 / 79.52。Table II（复杂物体手内迁移，Multi 多物体列）：从零 RL 1.00；预训练 actor 的 RL 40.00；动作空间残差 8.80；单尺度先验 65.80；TacBPM 冻结解码器 73.30；TacBPM 主设置 70.00。Table IV 真机指定轴旋转（每物体 6 个方向各 10 次，需沿指令方向转过 180° 以上且 20 s 内）：按表中分项求和，TacBPM 约 181/240，从零约 69/240，无触觉约 61/240。Table V 臂-手 Grasp-to-AnyPose 仿真（每物体 1024 次确定性评测，从零训练 / TacBPM）：小锤 77.73 / 97.56，蓝刷 27.44 / 95.70，订书钉标记笔 0.49 / 97.36，木槌 91.31 / 96.29；真机臂-手仅给出一次抓取-抬升-到位的定性演示（FoundationPose + 正运动学取关键点）。作者自陈局限：从球体先验外推有限（sphere 1.2 仅 13.30%）、已见尺度上触觉增益小且区间重叠、指令切换瞬态会掉物、真机每个物体-轴对只有 10 次、臂-手教师与先验单独训练。
- **对 wiki 的映射：** [paper-tacbpm](../../wiki/entities/paper-tacbpm.md)

## 对 wiki 的映射

- [TacBPM 论文实体](../../wiki/entities/paper-tacbpm.md)
- [Sharpa Robotics](../../wiki/entities/sharpa-robotics.md)
