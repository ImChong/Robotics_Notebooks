# Do as I Do: Dexterous Manipulation Data from Everyday Human Videos（arXiv:2606.19333）

> 来源归档

- **标题：** Do as I Do: Dexterous Manipulation Data from Everyday Human Videos
- **arXiv：** <https://arxiv.org/abs/2606.19333>
- **PDF：** <https://arxiv.org/pdf/2606.19333>
- **v1 日期：** 2026-06-17（UTC 17:57:34）；摘要页仅列 v1
- **分类：** cs.RO（主）；cs.CV；摘要页备注仅含项目页链接，未标注会议或期刊
- **作者与机构（arXiv 摘要页 + HTML 全文页署名）：** Bhawna Paliwal\*、Haritheja Etukuru\*、William Liang\*（\* 为 equal contribution）、Pieter Abbeel、Nur Muhammad "Mahi" Shafiullah、Jitendra Malik；机构 **UC Berkeley**。全文致谢注明 Abbeel 同时是 Amazon Scholar，但该工作完成于 UC Berkeley、与 Amazon 无关。摘要页不列机构，机构以 HTML 全文署名为准。
- **项目页：** <https://do-as-i-do.com/>（页面含摘要、亮点、结果、方法四个锚点区块）
- **代码：** <https://github.com/malik-group/do-as-i-do>（项目页头部 Code 链接；2026-10-16 复核：仓库公开，MIT 许可）
- **开源状态：** **已开源（仅代码）**——仓库含 `reconstruction/`、`retargeting/`、`deployment/` 三个目录与 git 子模块；README 未提及数据集或模型权重发布；Sharpa Wave 手部 SDK 为专有，未随仓库提供（见 `deployment/README.md`）。
- **入库日期：** 2026-10-16
- **索引来源：** [wechat_sharpa_dexterous_manipulation_2026-10-16.md](../blogs/wechat_sharpa_dexterous_manipulation_2026-10-16.md)

## 一句话说明

从单目 RGB 人类视频（第一/第三人称、互联网、生成视频）重建 4D 手–物交互：HaWoR 追手、SAM 3D 出物体网格并以引导扩散做逐帧位姿跟踪；再在 MuJoCo Warp 里用采样式优化（SPIDER 基线 + warmup、随机力扰动、转换奖励三个改进）重定向为灵巧手可执行动作，并在 UR3e 双臂 + Sharpa Wave 手上回放 10 个动作。

## 文章转述（标注：文章转述）

- **文章转述：** 伯克利 Jitendra Malik 与 Pieter Abbeel 团队的 Do as I Do，从日常人类视频中重建手–物交互并重定向为灵巧手动作；文章把它归为「自上而下」路线（与 Dex-X 同类的「视频→仿真→策略」思路），并说明它基于 Sharpa 硬件（他队对照，不是 Sharpa 作者论文）。
- **文章转述：** 在 655 条野外视频重建结果上，重定向成功率从此前方法的 25% 提升到 71%。
- **文章转述：** 在 Sharpa Wave 上完成搅拌、倾倒、擦除等 10 项真机任务。
- **文章转述：** 张凯峰把 Malik 团队的工作归为自上而下，把 Bohg 与 Liu 团队的工作归为自下而上，与 Sharpa 自己的探索互相补位。

## 核心摘录

1. **（arXiv 摘要）目标：** 提出 Do as I Do，一种把单目 RGB 人类视频重建并重定向到多指灵巧机器人手的算法；可处理第一人称与第三人称的野外视频来源，输出机器人可执行动作序列，生成「机器人完备」的操作数据。
2. **（arXiv 摘要）结论与附带产出：** 作者称在有真值的数据集与线上视频片段集上，其手–物交互估计与操作轨迹抽取优于既有方法，并据实验提出一份面向人类数据采集者的「efficacy playbook」。
3. **（项目页）方法三步：** 重建（基础视觉模型 + 引导扩散追踪 3D 物体，深度估计对齐手与物体）→ 重定向（采样式优化，含 warmup 步、随机力扰动、转换奖励）→ 在 MuJoCo 中物理仿真；项目页称其「outperforms previous state of the art」，并提到 10 个真实世界动作；项目页不含 655 条、25%/71% 等具体数字，也未点名机器人手型号。
4. **（文章转述）关键数字：** 655 条野外视频重建结果上重定向成功率 25%→71%；10 项真机任务。（论文表 3 核对见下节：25% 是 SPIDER 退火采样基线，71% 是加入三个组件的完整方法。）

## 论文正文要点（标注：论文正文，自报）

- **重建（Sec 3.1）：** HaWoR 追手；SAM 3 分割；MoGe 深度与内参；SAM 3D 由单帧生成物体网格，并被改造为基于引导 flow-matching 扩散的跟踪器（形状锚定到规范网格，位姿朝上一帧引导；每帧采 25 个候选位姿，用聚类按加权 SE(3) 距离选择，论文称与对数似然选择相当而最多快 30 倍）。
- **重定向（Sec 3.2）：** MuJoCo Warp，时间步 0.005 s；MPPI 式采样优化，基于 SPIDER；每 0.5 s 规划一次、3 s 视野、1024 个样本、32 次迭代；运动学参考用 mink 的指尖匹配；物体用 CoACD 凸分解并外扩 2 mm。成功判据：平均位置误差小于 0.1 m 且平均旋转误差小于 0.5 rad。
- **重建结果（Table 2）：** DexYCB 上 Ours F-5/F-10/CD 为 0.71/0.93/0.66，FoundationPose 为 0.69/0.89/0.89；HOI4D 上 Ours 为 0.72/0.91/0.49，FoundationPose 为 0.71/0.91/0.49。150 条野外视频的人评偏好：67% 选 Ours、18% 选 FoundationPose、15% 平局。
- **重定向结果（Table 3，成功率）：** 在 655 条重建参考上，Annealed Sampling（SPIDER 基线）0.25；加 Warmup 0.66；再加 Perturbation 0.67；再加 Transition Reward 0.71。在 OakInk2（1,352 条干净 MoCap 双手轨迹）上依次为 0.72、0.77、0.79、0.81。
- **真机部署（Sec 4.4）：** 流水线共产出 500 条人工核验轨迹（互联网 53%、第一人称 31%、生成视频 16%）；真机展示 10 个动作（whisking、pouring、dusting、squeezing、tamping、erasing、stirring、hammering、spreading、picking），覆盖 tripod、power、ventral、parallel extension 等抓型；平台为 UR3e 双臂 + Sharpa Wave（22 DoF），均 50 Hz 下发；论文未给出这 10 个动作的真机成功率，仅为定性展示（胶片图与项目页视频）；重建结果的初始位姿（x、y、z、yaw）需手动对齐机器人工作区。
- **数据过滤 playbook（Sec 4.5，100DOH）：** 2,000 个 10 秒片段中仅 187 个（9%）含有意义的手–物交互，最终仅 83 个（4%）通过重建质量检查，作者据此称不做过滤有约 20 倍代价。
- **作者自述局限（Sec 5）：** 假设刚体与较准的单目度量深度；单目下难分「真接触」与「遮挡」；只重建手与单个物体，不建模场景与铰接；仿真动力学仅近似，限制真机上限。

## 对 wiki 的映射

[paper-do-as-i-do](../../wiki/entities/paper-do-as-i-do.md)
