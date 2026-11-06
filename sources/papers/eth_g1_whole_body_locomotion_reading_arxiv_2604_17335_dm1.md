# Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking 论文通读

> 来源归档（原文通读 · 支撑 `wiki/methods/diffusion-motion-generation.md`）

- **标题：** Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking
- **作者：** Zewei Zhang, Kehan Wen, Michael Xu, Junzhe He, Chenhao Li, Takahiro Miki, Clemens Schwarke, Chong Zhang, Xue Bin Peng, Marco Hutter（ETH RSL / SFU）
- **类型：** paper / humanoid / diffusion / motion-tracking / perceptive-locomotion
- **原文：** <https://arxiv.org/abs/2604.17335>（2026-11 通过 arXiv PDF v2 获取全文，IEEE RA-L 2026 录用版）
- **一句话说明：** 扩散运动生成器（MDM 架构）在线产出地形感知参考，RL 跟踪器在冻结生成器下闭环微调，Unitree G1 机载感知完成箱攀、跨栏、楼梯与混合地形。
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 方法（数据构建与增广、预训练、RL 微调、部署细节）、§IV 实验（硬件结果、Table III 在线生成对比、Fig. 4 微调对比）、§V 结论与局限；Table I 奖励项已核对；参考文献列表仅浏览，未逐篇阅读被引文献（含 MDM、PARC、CLoSD）。项目页与补充视频未阅读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（摘要/§I）：** 纯奖励塑形 RL 易得到下肢主导行为；纯运动跟踪只能回放固定参考；直接部署生成的运动学轨迹可能出现足滑或时间不连续。
- **数据（§III-A）：** 初始约 5 分钟动作（视频经 GVHMR 重建，接触约束 IK 重定向，另用公开数据集），增广后约 1 小时（箱 35–75 cm、栏 25–45 cm、台阶 15–20 cm、随机小箱行走）；不直接用重定向轨迹，而是用 DeepMimic 式跟踪策略回放得到物理可行轨迹。
- **跟踪器（§III-B1）：** IsaacLab + PPO，单卡 RTX 4090；奖励 R_pre = r_mimic + r_reg；动作为 23 维目标关节位置；观测含参考状态、5 帧本体感受与 17×11 高程扫描。
- **生成器（§III-B2）：** 沿用 MDM 架构，预测 0.5 s（25 帧）的根位置/朝向、23 关节位置、身体连杆位置；条件为目标朝向向量、高程扫描与过去两帧运动特征；重建损失 + 速度、关节一致、地形穿透几何损失；训练时扰动高程与历史状态。
- **闭环微调（§III-C）：** 生成器冻结，以机器人过去两帧状态为条件（非自回归反馈自身预测）；仅 2 步去噪；对生成与观测注入噪声；随机化朝向并加入朝向跟踪奖励 R_post = r_mimic + r_reg + r_task；地形扩展到台阶 15–25 cm、连续栏 25–55 cm、箱/金字塔 30–85 cm。微调后的跟踪器相当于"运动滤波器"。
- **部署（§III-D）：** 全机载；DLIO 位姿 + Elevation Mapping CuPy 高程；被动颈部造成的俯仰用躯干 IMU 补偿；TensorRT 推理约 0.02 s；训练视界 0.5 s，部署每 0.25 s receding-horizon 更新参考；生成器在 Jetson Thor，跟踪器在 Jetson Orin。
- **硬件结果（§IV-A）：** 75 cm 箱的三种上下方式、阶梯、连续栏、混合地形；未见过的地形组合（如楼梯后接跳下）也能衔接；观察到绕箱侧向通过的局部导航行为。
- **在线生成收益（§IV-B1，Table III，仿真 500 机器人）：** 箱攀均值 0.859→0.987，跨栏 0.805→0.990，上楼梯 0.845→0.997（固定参考 Tracker Only → Tracker+Gen.）；箱高 80 cm 处 0.230→0.962，台阶 25 cm 处 0.114→0.986。
- **微调收益（§IV-B2，Fig. 4）：** 五类任务（上/下箱、跨栏、上/下楼梯）微调后成功率均更高，难度越大增益越大；作者解释为预训练跟踪器只见过平滑的离线参考，与生成器在线参考存在分布偏移。
- **局限（§V）：** 依赖 LiDAR 高程建图，建图质量下降时性能明显恶化；未涉及 loco-manipulation 与室外。
