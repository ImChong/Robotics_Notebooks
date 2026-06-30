# GRAIL（arXiv:2606.05160）

> 来源归档（ingest · 项目页 + arXiv 全文 + 官方代码仓）

- **标题：** GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors
- **缩写：** **GRAIL**
- **类型：** paper / humanoid loco-manipulation / synthetic data / sim-to-real
- **arXiv：** <https://arxiv.org/abs/2606.05160>（HTML：<https://arxiv.org/html/2606.05160v1>）
- **PDF：** <https://arxiv.org/pdf/2606.05160>
- **项目页：** <https://research.nvidia.com/labs/dair/grail/>
- **代码：** <https://github.com/NVlabs/GRAIL>
- **作者：** Tianyi Xie*, Haotian Zhang*, Jinhyung Park*, Zi Wang*, Bowen Wen, Jiefeng Li, Xueting Li, Qingwei Ben, Haoyang Weng, Yufei Ye, David Minor, Tingwu Wang, Chenfanfu Jiang, Sanja Fidler, Jan Kautz, Linxi Fan, Yuke Zhu, Zhengyi Luo‡, Umar Iqbal‡, Ye Yuan‡（*co-first；‡project leads）
- **机构：** NVIDIA、UCLA
- **入库日期：** 2026-06-30
- **一句话说明：** 全数字人形 loco-manipulation 数据生成管线：在已知 3D 场景/相机/尺度/机器人比例角色下，用视频基础模型合成交互视频，再重建 metric 4D HOI 轨迹、重定向到 Unitree G1 并训练任务通用 tracker；仅用生成数据训练 egocentric 视觉策略，真机 pick-up 84%、爬楼梯 90%。

## 核心摘录（策展，非全文）

1. **问题设定：** 人形 loco-manipulation 需要跨物体、全身动作与场景几何的机器人可执行示范，但遥操作与动捕难规模化；从野外视频重建 4D 轨迹又面临相机、尺度、形态与接触歧义。
   - **对 wiki 的映射：** [paper-grail](../../wiki/entities/paper-grail.md)、[Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md)

2. **资产条件 4D HOI 生成：** 用 Infinigen 等构建仿真就绪场景，Blender 渲染首帧（已知内外参），VLM 生成交互提示，VFM（如 Kling）合成静态相机 HOI 视频；再用 GENMO + WiLoR 估计人体/手部，FoundationPose 跟踪物体，联合优化关键点/投影/深度/接触损失，在特权 3D 配置下恢复 metric 4D 轨迹。
   - **对 wiki 的映射：** [paper-grail](../../wiki/entities/paper-grail.md)

3. **任务通用 tracker：** 重定向到 Unitree G1 后，在预训练全身控制器（SONIC）上训练两类互补 tracker——**物体感知 latent adaptor**（操作）与 **场景感知 height-map tracker**（地形/坐姿）；生成 **20,000+** 序列，覆盖 pick-up、全身操作、sitting、楼梯/坡道/路缘等。
   - **对 wiki 的映射：** [SONIC](../../wiki/methods/sonic-motion-tracking.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)

4. **Sim-to-real：** 仅用 GRAIL 生成数据训练 egocentric RGB 策略（VIRAL 管线 + 视觉域随机化与相机对齐），部署于 Unitree G1；pick-up **84%**、stair-climbing **90%** 真机成功率。另以 95% GRAIL + 5% 遥操作混合微调 GR00T，抓取成功率优于纯遥操作。
   - **对 wiki 的映射：** [paper-viral-humanoid-visual-sim2real](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md)、[loco-manip-161-148-gr00t-n1](../../wiki/entities/paper-loco-manip-161-148-gr00t-n1.md)

5. **策展地图坐标：** 同时出现在 [Loco-Manip 161 篇 #061](../../sources/papers/loco_manip_161_survey_061_grail.md)（03 视觉感知驱动）与 [运动小脑 64 篇 #57](../../sources/papers/motion_cerebellum_survey_57_grail.md)（H 真实任务）；知识库已合并为单一实体页。
   - **对 wiki 的映射：** [paper-grail](../../wiki/entities/paper-grail.md)

6. **公开数据集：** [PhysicalAI-Robotics-Locomanipulation-GRAIL](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) 发布 ~22k 条 G1 post-SONIC 轨迹（video + 4D recon + robot/objects pkl + USD，约 250 GB）。
   - **对 wiki 的映射：** [grail-locomanipulation-dataset](../../wiki/entities/grail-locomanipulation-dataset.md)

## 对 wiki 的映射

- 沉淀实体页：[paper-grail](../../wiki/entities/paper-grail.md)
- 公开数据集：[grail-locomanipulation-dataset](../../wiki/entities/grail-locomanipulation-dataset.md)
- Loco-Manip 分类 hub：[loco-manip-161-category-03-visuomotor](../../wiki/overview/loco-manip-161-category-03-visuomotor.md)
- 运动小脑分类 hub：[motion-cerebellum-category-08-real-tasks](../../wiki/overview/motion-cerebellum-category-08-real-tasks.md)
- 姊妹策展：[loco_manip_161_survey_061_grail.md](./loco_manip_161_survey_061_grail.md)、[motion_cerebellum_survey_57_grail.md](./motion_cerebellum_survey_57_grail.md)

## 参考来源（原始）

- 项目页：<https://research.nvidia.com/labs/dair/grail/>
- arXiv：<https://arxiv.org/abs/2606.05160>
- 代码：<https://github.com/NVlabs/GRAIL>
- 数据集：<https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL>
