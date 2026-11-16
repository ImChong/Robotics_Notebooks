# Large Video Planner Enables Generalizable Robot Control（arXiv:2512.15840）

> 来源归档（paper，全文通读）

- **标题：** Large Video Planner Enables Generalizable Robot Control
- **类型：** paper
- **作者：** Boyuan Chen、Tianyuan Zhang、Haoran Geng（共同一作）等；MIT / UC Berkeley / Harvard
- **arXiv：** <https://arxiv.org/abs/2512.15840>（本次读 v1，2025-12-17）
- **PDF：** <https://arxiv.org/pdf/2512.15840>
- **项目页：** <https://www.boyuan.space/large-video-planner/>
- **入库日期：** 2026-11-12
- **核对范围：** 读了 v1 PDF 全部正文（摘要、§1 引言、§2 相关工作、§3.1–3.3 方法、§4.1–4.3 评测、§5 局限、§6 结论）与附录 A–G（数据细节、训练超参、HaMeR/MegaSaM 对齐公式、夹爪重定向、任务与机器人设置、定性图）；参考文献仅扫过，未逐篇核对；图表以图注与表格文字为准，未逐帧看视频；未读项目页视频，未运行任何代码或模型。
- **通读：** 已完成
- **一句话说明：** 把 Wan 2.1 14B 视频模型用 Diffusion Forcing 与 History Guidance 在 LVP-1M（约 140 万条人类与机器人短视频）上继续预训练，生成零样本视频计划，再用 HaMeR + MegaSaM + Dex-Retargeting + cuRobo 把人手视频转成机器人动作（开环执行）。

## 核心摘录（均出自论文）

1. **范式（摘要、§1）：** 以大规模视频预训练为机器人基础模型主模态，替代「MLLM 加动作头」的 VLA；模型据一张观测图加文字指令生成视频计划，再做动作提取。
2. **模型（§3.1）：** 时间因果 3D VAE（16 通道潜变量）+ Diffusion Forcing Transformer，在 Wan 2.1 14B 权重上训练，flow matching；训练时随机取 0–6 个潜帧作历史，历史与未来各自独立噪声，历史 50% 概率零噪声（Fig.2）。采样用 History Guidance 与文本 CFG 的组合（式 1–4）；上下文最多 24 帧（6 潜帧），可迭代续写做多阶段计划（Fig.7）。
3. **训练（§3.1、附录 B）：** 两阶段：继续预训练 60k 步、batch 128、约 200B token；低相机运动子集微调 10k 步；共约 14 天、128 张 H100。单样本 49 帧、832×480；学习率 1e-5（预训练）与 2.5e-6（微调）。
4. **数据 LVP-1M（§3.2、Table 1、附录 A）：** 8 个来源（Bridge、DROID、Language Table、AgiBot-World；Ego4D、Epic-Kitchens、Something-Something、Panda-70M 过滤子集），共 1.4M 条片段、4.1M 条带重复的字幕，每条 2–5 种字幕；机器人片段按人类速度重采样；3 秒、16 fps。
5. **动作提取（§3.3、附录 D）：** HaMeR 逐帧 MANO 手部位姿，MegaSaM 提供深度与相机参数以反投影手腕并校正平移；SLERP 与因果 Savitzky-Golay 平滑；Dex-Retargeting 映射手指关节；固定旋转 M 对齐机器人控制系；cuRobo 解臂 IK；夹爪则用 GraspNet 预测抓取位姿加启发式触发（附录 D.6）。
6. **视频计划评测（§4.1–4.2、Table 2）：** 第三方众包约 200 条任务，筛选后 100 条（一张图加一句指令）；对比 Wan 2.1 I2V 14B、Cosmos-Predict 2 14B、Hunyuan I2V；每个提示每个方法生成 4 段，第三方标注者打四级分。Level 3（任务完成）平均：Ours 59.3%、Wan 39.3%、Hunyuan 13.5%、Cosmos 7.5%；Level 4：44.0%、20.5%、7.2%、2.5%；Level 1 Ours 87.3%、Wan 83.9%。
7. **真机评测（§4.3、Fig.8、附录 E–F）：** Franka Panda + 平行夹爪（15 Hz）、Unitree G1 + Inspire 灵巧手（5 Hz）。夹爪组 A（Pick Objects 5/10、Pick A into B 3/10、Open Drawer 2/10、Press Button 4/10；π0 为 3、1、1、0；OpenVLA 全 0）；OOD 组 C（Ours 4/10、2/10、6/10、1/10；π0 0、0、1、0；OpenVLA 全 0）；灵巧手组 B 共 8 项（如 Open Door 6/10、Wipe Table 8/10、Tear off Clear Tape 2/5），π0 与 OpenVLA 因不支持多指手未测。π0 与 OpenVLA 用公开 checkpoint 直接测，无微调。
8. **局限（§5）：** 单条视频计划在单张 A100 上需数分钟，无法实时；4D 重建与手部位姿模型会出错；重定向对某些灵巧手不充分，对平行夹爪因自由度低而困难；整体开环执行，不足以完成灵巧任务。
9. **开源声明（摘要、§6）：** 声称开源模型、数据与训练代码；论文正文未给出具体 GitHub 仓库地址，仅给项目页。

## 笔者核对备注（非论文自述）

- §4.3 正文描述的任务类别（in-hand rotation、pen writing、screwdriver insertion、block stacking、bottle relocation）与 Fig.8 / 附录 E 实际列出的任务不一致；以后者为准。
- 真机每项仅 5 或 10 次试验，未给置信区间或试验设置的随机化细节。
- §4.3 正文称 Table 8，实为 Fig.8（论文笔误）。

## 对 wiki 的映射

- [paper-sa-2512-15840-lvp-large-video-planner-enables-generalizable-ro](../../wiki/entities/paper-sa-2512-15840-lvp-large-video-planner-enables-generalizable-ro.md)
- 清单级旧摘录：[sun_awesome_wm_2512_15840_lvp-large-video-planner-enables-generali](sun_awesome_wm_2512_15840_lvp-large-video-planner-enables-generali.md)
