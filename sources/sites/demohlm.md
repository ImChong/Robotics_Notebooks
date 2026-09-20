# DemoHLM 项目页

> 来源归档

- **标题：** DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation
- **类型：** site
- **链接：** <https://beingbeyond.github.io/DemoHLM/>
- **机构：** 北京大学（PKU）、超越智能（BeingBeyond）
- **论文：** <https://arxiv.org/abs/2510.11258>
- **代码：** <https://github.com/BeingBeyond/DemoHLM>
- **入库日期：** 2026-09-20
- **再核日期：** 2026-09-20
- **一句话说明：** BeingBeyond 官方项目页：方法总览、十任务真机视频、BibTeX；页脚链到 arXiv 与 GitHub。

## 开源状态（步骤 2.5 核查）

| 资源 | 状态 |
|------|------|
| 项目页 / 视频 / BibTeX | **已发布** |
| GitHub 仓库 | **部分开源**：公开仓存在，但 **截至 2026-09-20 仅含 README 与 `docs/` 项目站镜像**（overview 图、演示 mp4），**无训练 / 数据生成 / 部署脚本** |
| 预训练权重 / 仿真环境包 | **未发布**（README 与项目页均未列 HF / Drive / release） |

## 页面要点

- **方法图：** 单条 VR 仿真示范 → 物体坐标系轨迹 → 三阶段数据生成 → BC 训练 → 真机闭环部署。
- **真机视频：** LiftBox / PressCube / PushCube / Handover / GraspCube 等 ego + 第三视角 rollout。
- **作者：** Fu* / Xie* / Xu / Xiong / Yuan / Lu §；PKU + BeingBeyond。

## 对 wiki 的映射

- [paper-loco-manip-161-136-demohlm.md](../../wiki/entities/paper-loco-manip-161-136-demohlm.md)
- [demohlm_arxiv_2510_11258.md](../papers/demohlm_arxiv_2510_11258.md)
- [demohlm.md](../repos/demohlm.md)
