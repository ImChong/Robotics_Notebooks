# PlayWorld: Learning Robot World Models from Autonomous Play（arXiv:2603.09030）

> 来源归档（paper）

- **标题：** PlayWorld: Learning Robot World Models from Autonomous Play（CoRL 2026 清单所写标题为 "...from Autonomous Play Data"）
- **类型：** paper
- **作者：** Tenny Yin, Zhiting Mei, Zhonghe Zheng, Miyu Yamane, David Wang, Jade Sceats, Samuel M. Bateman, Lihan Zha, Apurva Badithela, Ola Shorinwa, Anirudha Majumdar（普林斯顿大学）
- **arXiv：** <https://arxiv.org/abs/2603.09030>（本次读 v2，2026-03-11 标注）
- **PDF：** <https://arxiv.org/pdf/2603.09030v2>
- **项目页：** <https://robot-playworld.github.io>（论文首页给出；本次未访问）
- **入库日期：** 2026-11-12
- **一句话说明：** 用 VLM 提任务、VLA 执行的无监督机器人自玩数据（30 小时）微调 SVD 视频世界模型，并配课程学习；用于接触预测、策略评估与模型内 RL 微调。
- **核对范围：** 通读 arXiv v2 PDF 正文 §1–§6 全部与附录 A–D（含 Table 1–5、公式 1–12 与文字描述）；图中的曲线、散点与柱状具体数值只能从图读，未逐点读出；参考文献仅浏览、未逐条核对；未读项目页与视频，未见代码仓库。
- **通读：** 已完成

## 核心摘录

1. **问题（§1）：** 视频世界模型在接触丰富的操作中会出现物体复制、凭空出现/消失、不真实形变等幻觉；作者假设原因在训练数据（成功偏置的人类示教）。
2. **数据采集（§3.2，附录 A）：** 在 DROID 平台上，VLM 根据当前图像提出带扰动的自然语言指令，π0.5-DROID 作为执行器；关节限位安全过滤器，VLM 检测物体漂向可达边界则下发「移回中心」指令；支持最长 8 小时连续无人值守采集（含夜间）。
3. **模型（§3.3，附录 A）：** 沿用 Ctrl-World 的设置，Stable Video Diffusion 骨干，因子化时空注意力，逐帧动作条件，联合预测 3 个相机视角（192×320），先在 DROID 预训练；8×H200、batch 64 训练两天（§3.3）。
4. **课程学习（§3.3，附录 A）：** 用冻结 CLIP 编码，对少量人类成功示教做 K-means（每任务 K=5）得到「成功质心」，按 play 数据到最近质心的距离分 R=5 档，采样分布由 (0.5,0.3,0.1,0.05,0.05) 每 5000 步线性退火到 (0.1,0.2,0.2,0.25,0.25)。
5. **实验设置（§4.1）：** 三个物体集（碗/胡萝卜/北极熊；方块与立方体；毛巾）；机器人 play 总计 30 小时；基线为 6 小时人类示教（Human Demo）与 6 小时人类自由玩耍（Human Play）。
6. **接触预测（§4.3，Table 1）：** 500+ 片段，来自 20+ 个策略的 rollout，人工分为 6 类（成功、漏抓、滑动、打滑、形变、碰撞）；成功类各训练混合相近，失败类上 Robot Play 优于人类数据；Robot Play (Curriculum) 在多数类别 LPIPS/SSIM 最好。
7. **策略评估（§4.4，Fig. 7）：** 18 个策略（从零训练的扩散策略与微调的 π0），每策略真机 20 次、每个世界模型 50 次仿真；PlayWorld 预测成功率与真机成功率 Pearson 相关 0.8766。
8. **模型内 RL 微调（§4.5，附录 D，Fig. 9）：** 用 DSRL（冻结扩散策略，学习初始噪声的小策略）加基于进度差分的稠密奖励；起点为 <10 条示教的扩散策略；摘要与 §4.5 称真机成功率「提升最多 65%」。
9. **扩展（§4.6）：** 数据规模增至 30 小时时视觉指标仍在提升；物体多样性 0%→100% 对留出物体的 LPIPS/SSIM 持续改善（Fig. 11 表）。
10. **作者承认的局限（§6）：** 采集策略未优化样本效率、冗余；依赖足够强的 play 策略；幻觉未消除（开环 rollout、固定动作视野、控制模式不匹配）；尚未超出受控实验室环境。

## 对 wiki 的映射

- [paper-playworld-autonomous-play-data](../../wiki/entities/paper-playworld-autonomous-play-data.md)
- 原有来源（综述定位）：[wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25](../blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
