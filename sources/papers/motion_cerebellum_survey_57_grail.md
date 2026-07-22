# GRAIL 用3D资产和视频先验生成人形LocoManip数据

> 来源归档（ingest · 运动小脑 64 篇长文 第 57/64）

- **标题：** GRAIL 用3D资产和视频先验生成人形LocoManip数据
- **类型：** paper
- **运动小脑分类：** H 真实任务
- **机构：** 英伟达
- **项目页：** https://research.nvidia.com/labs/dair/grail/
- **arXiv：** https://arxiv.org/abs/2606.05160
- **代码：** https://github.com/NVlabs/GRAIL
- **入库日期：** 2026-06-18
- **复核日期：** 2026-07-22
- **一句话说明：** 任务数据：3D 资产和视频先验生成 Loco-Manip 数据。输入是移动操作任务、场景几何、物体状态和机器人模型；实现上用规划、仿真、生成式数据或自主探索产生手脚协同轨迹，再筛选成可训练示范；目标是补足人形 loco-manip 最缺的长程、接触丰富数据。

## 核心摘录（策展，非全文）

- **在动作小脑地图中的位置：** H 真实任务，编号 **57/64**。
- **公众号站位：** 任务数据：3D 资产和视频先验生成 Loco-Manip 数据
- **论文复核要点：** GRAIL 在已知 3D 资产、场景和相机条件下生成 HOI 视频，再用 GENMO / WiLoR / FoundationPose 与接触/深度优化重建 metric 4D HOI，重定向到 G1 并训练 task-general tracker。
- **开源状态：** 官方 NVlabs/GRAIL 已开放 Docker 与 `grail.pipelines.*` 入口；数据集发布在 Hugging Face，README 仍标注 manipulation dataset 待完整发布。

## 对 wiki 的映射

- [paper-grail](../../wiki/entities/paper-grail.md)
- [motion-cerebellum-category-08-real-tasks](../../wiki/overview/motion-cerebellum-category-08-real-tasks.md)
- [loco-manip-contact-category-03-generative-data](../../wiki/overview/loco-manip-contact-category-03-generative-data.md)
- [grail_nvlabs](../repos/grail_nvlabs.md)

## 参考来源（原始）

- 项目页：https://arxiv.org/abs/2606.05160v1
- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- 项目页：<https://research.nvidia.com/labs/dair/grail/>
- 代码：<https://github.com/NVlabs/GRAIL>
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
