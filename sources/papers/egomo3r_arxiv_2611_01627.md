# EgoMo3R: Joint Egocentric Motion and Scene Reconstruction（第一人称人体运动与三维场景联合重建）

> 原始论文资料归档（非论文全文转载）

- **标题：** EgoMo3R: Joint Egocentric Motion and Scene Reconstruction / EgoMo3R：第一人称人体运动与三维场景联合重建
- **作者：** Chaitanya Patel、Yusuke Kato、Rin Takahira、Kazuki Kozuka、Juan Carlos Niebles、Ehsan Adeli
- **机构：** 斯坦福大学（Stanford University）、松下美国研发公司（Panasonic R&D Company of America）、松下控股株式会社（Panasonic Holdings Corporation）
- **会议：** NeurIPS 2026
- **版本：** arXiv:2611.01627v1，2026-11-01
- **论文原文（全文 HTML）：** https://arxiv.org/html/2611.01627
- **论文 PDF：** https://arxiv.org/pdf/2611.01627
- **项目页：** https://chaitanya100100.github.io/EgoMo3R/
- **官方代码：** https://github.com/chaitanya100100/EgoMo3R
- **运动先验代码：** https://github.com/chaitanya100100/motion_diffusion
- **入库日期：** 2026-11-15
- **核对范围：** 阅读 arXiv v1 HTML 全文：摘要、§1–5、Appendix A–D（数据/评测、实现、运动先验分析、局限与未来工作）；结合官方项目页、README、run/optimize.py、run/run_all_stages.sh、config/stage1–4.yaml 与 model/motion_prior.py 核对代码和方法差异。
- **通读：** 已完成
- **沉淀到 wiki：** [EgoMo3R 论文实体](../../wiki/entities/paper-egomo3r.md)

## 摘录 1：动机与任务（§1、§3.1）

- **任务：** 给定普通第一人称头戴相机 RGB 视频，联合估计相机到世界坐标系的轨迹 C、基于单目相对深度及每帧尺度的三维场景 pointmaps，以及 SMPL-X 人体运动 M；输出统一在度量尺度、重力对齐且地面锚定的世界参考系中。
- **症结：** 常见 Ego 人体重建假设外部惯性 SLAM 已提供真实、度量且重力对齐的相机轨迹；普通动态 Ego 视频中相机旋转大、运动模糊、人体自身难见，传统视觉 SLAM 与直接前向 3D 重建均不稳。
- **核心反向约束：** 恢复的人体头部轨迹约束相机运动；人体动作隐含的重力方向、地面与脚部接触约束场景对齐，而优化过的相机又改进动作生成条件。
- **对 wiki 的映射：** [paper-egomo3r](../../wiki/entities/paper-egomo3r.md) 对应联合求解，不是直接机器人策略或动作重定向算法。

## 摘录 2：视觉几何与条件运动先验（§3.2–3.4）

- **视觉先验：** MoGe-2 输出相对深度 Dr 与初始米制尺度先验；RAFT 提供前后向 optical flow；CoTracker3 提供跨长时间 2D point tracks；对象/地面分割提供地面一致性约束。
- **优化变量：** 对每帧深度按尺度 S 缩放，而非将所有 3D 点作为独立优化变量；优化相机 C 与尺度 S，再回投影求世界 pointmaps。
- **六类损失：** 光流重投影 L_flow、长程跟踪 L_track、尺度先验 L_depth、地面-人体对齐 L_scene、相机-人体头部轨迹一致 L_egocam、条件扩散人体运动先验 L_motion；采用 MAD 过滤异常对应点。
- **关键设计：** 扩散模型条件采用以首帧归一化的 SE(3) 不变轨迹 A(C)_i=C_0^{-1} C_i，而不是要求先有准确的重力对齐轨迹；因此可在相机估计尚未收敛时工作。
- **优化策略：** 光流先 warm-start 相机，再交替固定 M 优化 C/S、固定 C/S 优化 M，降低直接联合梯度优化的塌缩风险。
- **对 wiki 的映射：** [EgoMo3R 的流程图与核心机制](../../wiki/entities/paper-egomo3r.md)。

## 摘录 3：评测与可审计结论（§4、Tables 1–5、Appendix A）

- **数据：** EgoExo4D 与 Nymeria，Project Aria 采集；统一去畸变为 512×512、f=150 px。主结果为 EgoExo4D 718 段、Nymeria 828 段，每段 8 秒；另报告 random、locomotion-heavy、rotation-heavy 子集。
- **场景重建（Table 1，metric mean accuracy，越低越好）：** EgoExo4D EgoMo3R=0.357 vs CUT3R=0.433；Nymeria EgoMo3R=0.247 vs CUT3R=0.346。场景 completion 的指标同样改善。
- **相机（Table 2，metric ATE）：** EgoExo4D EgoMo3R=0.226 vs CUT3R=0.415、ViPE=0.426；Nymeria EgoMo3R=0.325 vs ViPE=0.276（后者更好）。因此不能声称 EgoMo3R 在每套数据的全部相机指标均第一。
- **人体（Table 3，global MPJPE）：** EgoExo4D 0.271、Nymeria 0.319；在所列非 oracle 基线中较好，但 per-frame 对齐的 MPJPE-PA 和 Foot Contact (FC) 并非处处第一。EgoExo4D 的运动「真值」使用 UniEgoMotion motion fits，Nymeria 则有运动标注，应区分。
- **消融（Table 4/5，EgoExo4D locomotion 子集）：** 单用 flow ATE=0.604，加 tracks 变 0.361；直接联合 motion 的 MPJPE=0.456，交替优化变 0.320；追加第三轮还可改善部分指标，论文主表采用两轮。
- **对 wiki 的映射：** [paper-egomo3r](../../wiki/entities/paper-egomo3r.md) 明确指标单位、数据集与基线。

## 摘录 4：实现与局限（Appendix B–D）

- **实现：** RAFT 双向流；CoTracker3 每隔 8 帧在 35×35 网格播种；MoGe-2；Adam 起始学习率 0.01。运动先验基于 AMASS（而非在评测集 EgoExo4D / Nymeria 上训练）；用 1D UNet 做条件运动扩散。
- **性能成本：** 论文报告单张 NVIDIA L40S 对 8 秒片段约 8 分钟（含几何预计算），显著慢于前向方法；属于离线优化，不能将其当作实时人体追踪器。
- **失败模式：** 平滑纹理 / 手持物影响点跟踪、严重运动模糊、地面检测缺失、动态物体干扰；MAD 只能缓解部分异常。
- **运动边界：** 重点是 locomotion 场景。头部轨迹较能约束下肢与躯干，难以独立推断手腕与复杂人-物操作；不能将其描述为完整灵巧手操作重建系统。
- **论文与代码差异：** 论文表述为 2 轮场景/运动交替；公开运行脚本实现 stage1/3 场景优化、stage2/4 扩散动作采样，stage4 含轨迹/滑动 guidance。详见源码归档。
- **对 wiki 的映射：** [源码实现与复现边界](../../wiki/entities/paper-egomo3r.md)。

## 原始资料

- [arXiv v1 论文全文](https://arxiv.org/html/2611.01627)
- [论文官方项目主页](https://chaitanya100100.github.io/EgoMo3R/)
- [官方实现](https://github.com/chaitanya100100/EgoMo3R)
