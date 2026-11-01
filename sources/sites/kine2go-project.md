# Kine2Go 项目页

> 来源归档（ingest）

- **标题：** Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions
- **类型：** site / project-page
- **URL：** <https://nomagiclab.github.io/kine2go-pipeline/>
- **配套论文：** [arXiv:2606.14433](https://arxiv.org/abs/2606.14433) — 归档见 [论文来源](../papers/kine2go_arxiv_2606_14433.md)
- **代码：** <https://github.com/nomagiclab/kine2go-pipeline> — 归档见 [代码仓库来源](../repos/kine2go-pipeline.md)
- **数据：** <https://huggingface.co/datasets/MIMUW-Robotics/kine2go>
- **入库日期：** 2026-10-27

## 一句话摘要

华沙大学团队发布的 Go2 四足运动学数据集与生成管线，覆盖多来源动作重定向、Genesis 中 PPO 模仿和扰动 rollout 采集。

## 公开信息要点

- 目标平台为 Unitree Go2；管线将不同四足形态的动作转换为 Go2 兼容轨迹。
- 每条参考动作训练一个模仿策略，并从策略采集经筛选的仿真 rollout。
- 官方页链接论文、GitHub 代码与 Hugging Face 数据集。
- 数据许可随上游动作来源变化，数据集逐片段记录许可。

## 关联资料

- [论文归档](../papers/kine2go_arxiv_2606_14433.md)
- [代码仓库归档](../repos/kine2go-pipeline.md)
