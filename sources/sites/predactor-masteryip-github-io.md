# PredActor 项目页（masteryip.github.io/predactor.github.io）

> 来源归档（最近复核：2026-11-19）

- **类型：** site / project page
- **主页：** <https://masteryip.github.io/predactor.github.io/>
- **团队与联系：** <https://masteryip.github.io/predactor.github.io/#people>
- **论文：** [arXiv:2609.24840](https://arxiv.org/abs/2609.24840) — [论文归档](../papers/predactor_arxiv_2609_24840.md)
- **官方代码仓：** <https://github.com/MasterYip/PredActor> — [仓库归档](../repos/predactor.md)
- **官方工件：** <https://huggingface.co/MasterYip/PredActor_Artifacts> — [工件归档](../repos/predactor-artifacts.md)
- **机构/团队标识：** 哈尔滨工业大学、上海创新研究院、RoboParty Lab、清华大学、上海交通大学；项目页另展示 HexLab、SFTR 标识。
- **入库日期：** 2026-09-24
- **最近复核：** 2026-11-19
- **核对范围：** 项目页完整静态文本（`predactor_site.txt`）与 GitHub 仓库 Release Checklist 对照
- **一句话说明：** G1 机载联合状态–动作扩散控制（proprio-only，CG+CFG，50 Hz Jetson Orin NX）：展示文本指令切换、点导航、语义插值、外力抗扰及真机部署演示。

## 项目页要点

- 项目主页围绕方法原理、文本控制（Walk/Jog/Run/Squat 等）、摇杆与目标点跟随（Destination Follow）、语义插值（MotionCLIP latent interpolation）、300 N 抗扰与 Unitree G1 实机视频组织，并在 `#people` 公布核心作者（Yip 等，Hit / SII / RoboParty / THU / SJTU）联系入口。
- 与 2026-10-08 状态相比，GitHub 主仓现已进一步开放数据采集染色、BC 训练与 C++ G1 部署栈（仅 DAgger 仍未发布）。

## 开源核查（2026-11-19）

| 组件 | 状态 |
|------|------|
| 论文（v3）与项目演示 | **公开** |
| MuJoCo Web 评测代码 | **已开源** — `predactor-eval`（Python 3.10 / `uv sync --locked`） |
| 数据采集、染色与 BC 训练 | **已开源** — `PredActor/collection/` 与 `PredActor/train.py`（见 `docs/training.md`） |
| G1 LibTorch C++ 部署栈 | **已开源** — `PredActor/diffusion_deploy/`（支持 Unitree MuJoCo、有线 G1 与 Jetson 机载） |
| 官方模型工件 | **已发布** — HF `MasterYip/PredActor_Artifacts`（含 PDP051 ckpt、G1 MotionCLIP 及可选部署包） |
| DAgger 迭代精炼 | **未发布** — 官方 README Release Checklist 唯一未勾选项 |

## 对 wiki 的映射

- [PredActor 论文与项目实体](../../wiki/entities/paper-predactor.md)
- [官方代码仓归档](../repos/predactor.md)
- [Hugging Face checkpoint 归档](../repos/predactor-artifacts.md)
