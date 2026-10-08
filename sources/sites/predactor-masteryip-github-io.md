# PredActor 项目页（masteryip.github.io/predactor.github.io）

> 来源归档（最近复核：2026-10-08）

- **类型：** site / project page
- **主页：** <https://masteryip.github.io/predactor.github.io/>
- **团队与联系：** <https://masteryip.github.io/predactor.github.io/#people>
- **论文：** [arXiv:2609.24840](https://arxiv.org/abs/2609.24840)
- **官方评测代码：** <https://github.com/MasterYip/PredActor> — [仓库归档](../repos/predactor.md)
- **官方 checkpoint：** <https://huggingface.co/MasterYip/PredActor_Artifacts> — [工件归档](../repos/predactor-artifacts.md)
- **机构/团队标识：** 哈尔滨工业大学、上海创新研究院、RoboParty Lab、清华大学、上海交通大学；仓库 banner 另展示 HexLab、SFTR 标识。
- **入库日期：** 2026-09-24
- **最近复核：** 2026-10-08
- **一句话说明：** G1 机载 joint state–action diffusion；proprio-only；CG+CFG；50 Hz Orin NX；提供仿真/真机演示，并链接到当前可用的 MuJoCo 评测 release。

## 项目页要点

- 项目主页围绕方法、实验视频和团队信息组织；团队与联系入口见 `#people`。
- 论文报告 G1 Jetson Orin NX 50 Hz 闭环和 simulation / physical robot demonstrations；项目页提供文本控制、摇杆转向、外扰响应和语义插值示例。
- 代码仓目前提供公开 MuJoCo evaluator 与 PDP051 / MotionCLIP checkpoint；训练数据采集、BC/DAgger 训练和真机部署路径仍未公开。

## 开源核查（2026-10-08）

| 组件 | 状态 |
|------|------|
| 论文与项目演示 | **公开** |
| MuJoCo 评测代码 | **公开** — Linux + Python 3.10 / uv；启动本地浏览器评测 |
| 评测模型工件 | **公开** — PDP051 policy 与 G1 MotionCLIP encoder；数据集目录当前无数据 payload |
| 训练、采集与部署 | **未发布** — 官方 README release checklist 尚未勾选 |
| 许可 | 代码仓与 Hugging Face 模型卡标注 MIT；第三方资产/依赖遵循各自条款 |

## 对 wiki 的映射

- [PredActor 论文与项目实体](../../wiki/entities/paper-predactor.md)
- [官方代码仓归档](../repos/predactor.md)
- [Hugging Face checkpoint 归档](../repos/predactor-artifacts.md)