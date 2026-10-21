# WoVR+: World Model as Reliable Simulator for Versatile VLA Policy Optimization with RL（arXiv:2610.17922）

> 来源归档（ingest）

- **标题：** WoVR+: World Model as Reliable Simulator for Versatile VLA Policy Optimization with RL
- **arXiv：** <https://arxiv.org/abs/2610.17922>（PDF：<https://arxiv.org/pdf/2610.17922>；HTML：<https://arxiv.org/html/2610.17922v1>）
- **代码：** 论文可见内容与项目页均未给仓库（2026-10-17 核查）
- **项目页：** <https://youmo445.github.io/WoVR_Plus.github.io/>（归档见 [wovr-plus-project](../sites/wovr-plus-project.md)）
- **作者：** Shangqing Zhou、Zhennan Jiang、Ning Ma、Xin Fu、Yupeng Zheng、Shuai Tian、Huangrui Li、Jiahao Liu、Chao Yu、Zhongpu Xia、Haoran Li（通讯）、Dongbin Zhao；单位：中科院自动化所、中国科学院大学人工智能学院、中关村学院、西交利物浦大学、清华大学、Anyverse Dynamics（来自 arXiv HTML v1）
- **入库日期：** 2026-10-17
- **来源：** 具身智能小站 2026-10-17 公众号摘要 + arXiv HTML（前 100k 字符，附录后部未读）
- **一句话说明：** 在 WoVR 基础上扩展：VLA 对齐的多视角 + 本体感觉世界模型、动作块级共享噪声、阶段局部想象训练，用世界模型当可靠模拟器做 VLA 的 RL。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 仅演示视频与方法说明，页面样式取自 RISE；无 Code / 数据链接，未标注 coming soon。
- **论文核查：** 可见部分无代码可用性声明（附录后部未读，未核实）。
- **结论：** **未列链接**（截至 2026-10-17）。

## 摘录 1：方法

- 世界模型：Wan2.2-TI2V-5B 视频骨干 + VACE 式并行 DiT 控制分支；绝对关节目标经正运动学变成末端位姿并投影为逐视角动作图；射线条件；首帧锚点 + 稀疏近期历史（C=5，stride 4）；5 步积分，DMD2 蒸馏可单步。
- RL：对想象 rollout 做 clipped GRPO，Flow-SDE 取自 πRL，基于 RLinf；真机策略为 π0.5。
- 块级噪声：一个高斯噪声向量在整个动作块内共享，只改时间相关性，保持逐 token 边缘噪声尺度。
- 阶段局部训练：从有数据支持的阶段入口状态起算，每阶段有二值完成奖励与局部视野，阶段均匀加权。

**对 wiki 的映射：** 核心原理与流程总览。

## 摘录 2：结果

- RoboTwin 多视角（固定 / 随机初始）：Adjust Bottle SFT 79.00 / 70.70，WoVR+ 93.00 / 91.80；Click Bell SFT 71.00 / 72.66，WoVR+ 92.00 / 93.75；直接 RL（收敛）95.00 / 99.00（固定初始）更高。
- 视频预测（5 步，头部视角）：WoVR+ LPIPS 0.148、FVD 62.16、PSNR 22.28，约 9.4 FPS；单步 DMD 约 15 FPS。
- 真机六任务各 20 次：平均 39.2% → 55.8%（50 次 WM-RL 更新）。
- 手机包装四阶段：SFT 30%，阶段重置均匀采样 75%（15/20），到达匹配采样 20%。
- 世界模型与真机成功率相关：Pearson r=0.97，Spearman ρ=0.94（六个任务-策略对）。

**对 wiki 的映射：** 实验与评测、结论。

## 摘录 3：局限（结论段）

依赖预定义阶段与有数据支持的入口状态（人工标注）；块级 Flow-SDE 引入跨 token 相关但 GRPO 用逐 token 代理比率；真机每任务仅 20 次。

## 建议 wiki 动作

- 新建 [`wiki/entities/paper-wovr-plus.md`](../../wiki/entities/paper-wovr-plus.md)，并与已有前作索引页互链。
