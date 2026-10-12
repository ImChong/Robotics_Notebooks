# StereoEngine 论文来源归档

## 元数据

- **题目：** StereoEngine: Scaling Stereo Bimanual Robot Data without Stereo Cameras
- **作者：** Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, Daniel Seita
- **单位：** University of Southern California，Thomas Lord Department of Computer Science
- **预印本：** https://arxiv.org/abs/2610.12734
- **HTML：** https://arxiv.org/html/2610.12734v1
- **版本：** v1，2026-10-08，cs.RO
- **项目页：** https://stereoengine.github.io （归档见 sites/stereoengine-project.md）
- **代码：** 论文称 “will also release our code” 并提及数据集发布，但未给链接；截至 2026-10-12 未找到官方仓库。
- **读取范围：** arXiv 摘要页与 HTML 全文抽取，附录细节读到的部分已写入；训练超参未报告。

## 论文要解决的问题

缺少大规模真实双目数据，导致基于双目的操作策略（含 VLA）难以训练；现有机器人数据集绝大多数是单目。论文给出两条合成双目数据的途径，不需要双目相机。

## 方法与训练信息

1. **StereoEngine-L（单目→双目）：** 左目视频由 Video Depth Anything 估逐帧逆深度，缩放使视差中位数 δ₀ = 63 px（约 7.6% 帧宽）；用 softmax 前向 splatting（权重 √2^d，视差大的遮挡更近的像素）把左视图 warp 到右视图，空洞形成遮挡掩码并略膨胀；再用微调的 Wan2.2-VACE-Fun 补全右视图（warp 视频与掩码同时作为填充区域和控制输入，附文本提示）。
2. **L 的训练：** 不使用任何真实双目对；用 HuggingFace 上 3,469 条 PiPER 机械臂单目 episode，掩码来自同一前向 warp，δ₀ 在 35–95 px 间均匀采样，部分机械臂掩码教模型补出机械臂，10% 的片段做视差 dropout（0.05–0.25）。训练超参未报告。
3. **StereoEngine-S（仿真联合生成）：** 在 RoboTwin 中以所选基线放置第二虚拟相机渲染真双目；两个视图转 Canny 边缘图并拼成单个视频，再加两张真实参考图（工作区双目照片）和文本提示，交给 Wan2.2-Fun-A14B-Control 零样本联合生成两视图。
4. **硬件与策略：** 两只 AgileX PiPER 臂，ZED Mini 2 双目相机，Meta Quest 2 遥操作；策略为基于 π0.5 的双目条件 ExStereo（L、S 均用）与 StereoVLA（仅 S）。
5. **L 设置：** 3,469 条中期训练 episode 后在真实遥操作数据上微调：Stack Two Bowls 200 条、Lift Transparent Bottles 50 条、Close Door 20 条。
6. **S 设置（无真实示范）：** Stack Two Bowls 200 条合成、Lift Two Bottles 50 条、Container Plate 50 条。

## 论文报告的结果

- **立体质量（50 条保留真实片段，PSNR / SSIM / LPIPS / CTC）：** StereoCrafter 10.50 / 0.4574 / 0.6211 / 0.9807；SVG 12.83 / 0.5137 / 0.5128 / 0.9752；StereoEngine 15.31 / 0.5304 / 0.4696 / 0.9857（PSNR 比 SVG 高 2.48 dB）。
- **L 策略成功（ExStereo，每任务 20 次，无中期训练 / SVG / StereoCrafter / StereoEngine）：** Stack Two Bowls 9 / 7 / 10 / 13；Lift Transparent Bottles 11 / 10 / 13 / 18；Close Door 16 / 14 / 17 / 19。60 次合计：StereoEngine 50（83.3%），StereoCrafter 40（66.7%），无中期训练 36（60.0%），SVG 31（51.7%）。
- **S 策略成功（20 次/任务）：** 点云基线所有任务 0/20。ExStereo 用 StereoEngine 为 11/20（Stack Two Bowls）、13/20（Lift Two Bottles）、15/20（Container Plate）；StereoVLA 为 7/20、9/20、11/20。

## 复现与解释边界

- 论文承认：视频扩散对透明和高反光物体不稳；L 继承单目深度误差，右视图可能几何不一致，补全质量依赖微调数据量。
- S 需要逐场景精心设计提示与信息充分的参考图；Canny 边缘在相机更远时更嘈杂（周报所说“部分场景仍需人工设计提示”对应此点）。
- 范围缺口：每方法仅 3 个任务；S 没做 Lift Transparent Bottles 与 Close Door；Geometric Reciprocity、StereoPolicy 因无公开代码未对比；ExStereo 在发表时未公开，作者直接分享了实现。
- 项目页署名为 “Anonymous Authors”，没有代码或数据集链接。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-stereoengine.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 任务入口：wiki/tasks/bimanual-manipulation.md
- 方法入口：wiki/methods/stereo-matching-foundation-models.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.12734>
- arXiv HTML：<https://arxiv.org/html/2610.12734v1>
- 项目页：<https://stereoengine.github.io/>
