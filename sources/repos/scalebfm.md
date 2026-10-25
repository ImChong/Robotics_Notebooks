# ScaleBFM

- **URL：** <https://github.com/zengweishuai/ScaleBFM>
- **Maintainer：** Weishuai Zeng（上海人工智能实验室等，与 arXiv:2607.15163 作者团队一致）
- **入库日期：** 2026-07-18
- **最近核查：** 2026-10-25
- **关联论文：** [Scaling Behavior Foundation Model for Humanoid Robots](https://arxiv.org/abs/2607.15163)（arXiv:2607.15163）
- **项目页：** <https://scalebfm.github.io/>
- **开源状态（截至 2026-10-25）：** **已开源** — 上游仓库现包含 ScaleRetarget、ScaleTrack 与 ScaleBridge 的代码及使用说明。官方项目页的 Code 链接指向本上游仓库。另有 fork [Hxxxz0/ScaleBFM_Plus](./scalebfm-plus.md) 增加 SuperMimic-style 训练、checkpoint 与 ONNX 导出记录。
- **Tags：** #bfm #behavior-foundation-model #humanoid #motion-tracking #scaling #humanoid-transformer #isaaclab #unitree-g1

## 一句话说明

arXiv:2607.15163 **ScaleBFM** 官方实现仓库，现已公开动作重定向、IsaacLab 预训练、MuJoCo 评估与人形机器人部署代码。

## 仓库要点（2026-10-25 抓取）

| 字段 | 内容 |
|------|------|
| 创建时间 | 2026-07-16 |
| 默认分支 | `main` |
| 当前内容 | `ScaleRetarget/`、`ScaleTrack/`、`ScaleBridge/` 与 README 使用指南 |
| 许可证 | GitHub 仓库元数据未声明 |

## 对 wiki 的映射

- [ScaleBFM 论文实体](../../wiki/entities/paper-scaling-bfm-humanoid.md)
- [BFM 概念页](../../wiki/concepts/behavior-foundation-model.md)
- [Unitree G1](../../wiki/entities/unitree-g1.md)

## 维护备注

- 项目页 <https://scalebfm.github.io/> 页头 **Code** 指向本上游仓库。
- fork [ScaleBFM_Plus](./scalebfm-plus.md) 记录上游之外的 SuperMimic-style 训练与 ONNX/checkpoint 进展；其自报指标不应写成论文原文结果。
- 与 [bfm4humanoid/bfm4humanoid.github.io](https://github.com/bfm4humanoid/bfm4humanoid.github.io)（arXiv:2509.13780 网站仓）为 **不同论文/不同仓库**，勿合并归档。
