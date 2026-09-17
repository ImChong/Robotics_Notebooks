# WholeBodyWAM · UniMotion-4K 项目页

> 来源归档（site）

- **标题：** WholeBodyWAM: Learning Whole-Body World Action Models with Scalable Motion Priors
- **类型：** site
- **链接：** <https://zbzyjya.github.io/WholeBodyWAM/>
- **arXiv：** <https://arxiv.org/abs/2609.18197>
- **机构：** 南开大学；北京人形机器人创新中心；北京理工大学；清华大学
- **入库日期：** 2026-09-17
- **一句话说明：** UniMotion-4K 全身 motion prior + 两阶段 Video–Motion–Action WAM；天工 3.0 六项真机 whole-body 任务。
- **沉淀到 wiki：** [`wiki/entities/paper-wholebodywam-unimotion-4k.md`](../../wiki/entities/paper-wholebodywam-unimotion-4k.md)

## 开源状态

- **待发布**（步骤 2.5 核查，2026-09-17）。
- 导航栏 Code 图标标注 **Coming Soon**；无 GitHub / 数据集公开链接。

## 项目页要点

- **数据集 UniMotion-4K：** 11 源（互联网/ego 人视频、3D motion、多平台人形采集）→ **4.1K+ h** 全身运动。
- **表示：** 共享 **63D** root-free（21 关节局部旋转）；排除 global root / shape。
- **Stage I：** Motion Expert flow matching，语言条件 future body dynamics。
- **Stage II：** Video + Motion + Action Experts，MoT 层间注意力；真机臂/头/腰 direct，腿经 RL WBC。
- **真机任务：** Toy Pickup、Laundry Loading、Pillow Transfer、Kneeling Toy Storage、Toy Transfer、Box Transfer。
- **对照：** GR00T N1.7、FastWAM、τ₀-WM（+Motion Expert 可提升至 62.7%）。
- **推理：** 363 ms @ A100；闭环 replan，不 decode 未来视频。
