# ViDAL 论文来源归档

## 元数据

- **题目：** ViDAL: A Visual Dynamics-Grounded Action Latent Space for Vision-Language-Action Models
- **方法：** ViDAL（Visual Dynamics-grounded Action Latent）
- **作者：** Yuan Xu, Yixiang Chen, Qisen Ma, Jiabing Yang, Peiyan Li, Kai Wang, Jianhua Yang, Jianlou Si, Jun Huang, Jing Liu, Nianfeng Liu, Yan Huang, Liang Wang
- **单位：** arXiv 摘要页未列出机构（周报写作“中国科学院相关团队等，待核实”，未核实）
- **预印本：** https://arxiv.org/abs/2610.08150
- **HTML：** https://arxiv.org/html/2610.08150v1
- **版本：** v1，2026-10-06，cs.RO
- **项目页 / 代码：** arXiv 页面与正文均未给出；2026-10-12 核查无官方链接

## 论文要解决的问题

现有 VLA 的动作表示（离散 token、连续块、扩散/流匹配目标）主要拟合示范轨迹，对「动作造成了怎样的视觉变化」没有显式约束。ViDAL 把动作潜空间锚定到未来场景动态，使潜变量既能重构动作块，又携带动作引起的视觉变化信息。

## 方法与训练信息

- **Action VAE：** 归一化动作块 → 高斯后验连续潜变量，时间维 4× 下采样（T=H/4），潜维 D；附录描述编码器为带残差块和步幅下采样的时序卷积网络，解码器镜像上采样。下游取后验均值作确定性目标。
- **配置（附录 B）：** LIBERO / Franka：动作维 7，块长 H=32，D=8，观测 33 帧，128×128，50k 步；RoboTwin 2.0 / ARX：动作维 14，H=48，D=16，49 帧，96×128，50k 步。
- **视觉动力学对齐：** 冻结的 Wan VAE 编码观测视频片段，首帧 token 为 z₀，其余为未来目标 z_f；轻量预测器 P_θ 以 z₀ 与动作潜变量 z_a 通过交叉注意力预测 ẑ_f，训练后丢弃。
- **损失：** L = L_rec（动作 L1）+ λ_freq·L_freq（DCT 系数 L1，沿用 FAST）+ λ_KL·L_KL + λ_dyna·L_dyna（ẑ_f 对 stop-gradient 的 z_f 的 L1）；λ_dyna=1，λ_freq=1，λ_KL=1e-4。
- **接入 VLA：** 回归式专家 OpenVLA-OFT（LIBERO）与 StarVLA-OFT（RoboTwin）直接回归潜变量；流匹配专家 π0.5（两基准）以潜变量为条件流的终点；推理时用冻结 Action VAE 解码器还原动作块；VLM 主干不变；π0.5 的动作专家学习率设为 VLM 的 10 倍。
- **可选视频预测：** ẑ_f 经冻结视频 VAE 解码得到未来帧，仅作分析，不参与推理。
- **训练阶段：** 先按基准/平台分别训练 Action VAE 并冻结，再多任务训练 VLA 预测潜变量。
- **数据：** RoboTwin 2.0 为 50 任务 × 50 条 Clean 演示 = 2,500 条；LIBERO 用官方演示；Franka 4 任务各 50 条，ARX 3 任务各 100 条。

## 论文报告的结果

LIBERO 成功率（%，Object/Spatial/Goal/Long/Avg）：OpenVLA-OFT 98.0/97.7/96.1/95.3/96.8；+ViDAL 97.8/98.6/96.2/95.6/97.1（+0.3）；π0.5 98.2/98.8/98.0/92.4/96.9；+ViDAL 99.4/99.2/98.2/95.6/98.1（+1.2）。其他基线均值：π0-FAST 94.2、FASTer 97.9、GR00T-N1.5 93.9、WorldVLA 81.8、DreamVLA 92.6、UniVLA 95.2、Villa-X 90.1。

RoboTwin 2.0（50 任务平均，Clean / Random，每任务每设置 100 次 rollout）：π0 46.4 / 16.3；StarVLA-OFT 50.4 / 2.2 → +ViDAL 54.6 / 4.0；π0.5 54.3 / 33.2 → +ViDAL 65.5 / 43.1（+11.2 / +9.9）。论文称多任务 π0.5+ViDAL 超过官方单任务 π0。

真机（每任务 10 次 rollout）：Franka Research 3 + ZED 2i，4 任务（Place Mug on Plate、Stack Bowls、Place Bowl Close Drawer、Place Two Mugs in Basket）平均 +20.0%；ARX 双臂 + 三个 RealSense D405，3 任务（Fold Towel、Wipe Plate、Stack Bowls）平均 +23.4%；逐任务数字只在图 6，文中未给。

消融（π0.5，RoboTwin Clean / Random）：朴素 Action VAE 57.4 / 38.3；去 L_dyna 62.2 / 40.1；去 L_freq 63.1 / 40.4；去 L_KL 63.2 / 39.7；完整 65.5 / 43.1。视觉编码器：Wan VAE 最强，CogVideoX、SVD、V-JEPA 增益较小；稠密 L1 优于 InfoNCE，两者合用反而变差；只预测潜变量最好，raw-only 回到基线，联合头居中；λ_KL 最敏感。

潜空间几何：第一主成分方差占比 40.9%（原始动作）对 15.6%（ViDAL）；达 95% 方差所需成分 40 对 14；participation ratio 4.7 → 14.6。

## 复现与解释边界

- LIBERO 增益小（+0.3 / +1.2），源于基线接近饱和；RoboTwin Random 绝对值仍低（最好 43.1%，StarVLA-OFT 仅 4.0%）。
- 真机 3–4 个任务、每任务 10 次，统计功效有限。
- 完整对比与消融主要基于 π0.5；跨本体需重训 Action VAE；利用人类第一视角视频是作者列的未来工作。
- 论文无单独局限小节；以上为结合正文的读法。
- 无官方代码、项目页、权重（2026-10-12 核查）。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-vidal-visual-dynamics-action-latent-vla.md
- 方法入口：wiki/methods/vla.md
- 任务入口：wiki/tasks/manipulation.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.08150>
- arXiv HTML 全文：<https://arxiv.org/html/2610.08150v1>
