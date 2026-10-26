# ViT-H 官方实现仓库（mileret/ViT-H-release）

> 来源归档（ingest）

- **类型：** repo / humanoid control / video-to-humanoid
- **主仓库：** <https://github.com/mileret/ViT-H-release>
- **项目页：** <https://awfuact.github.io/vit-h/> — [项目页归档](../sites/vit-h.md)
- **论文：** <https://arxiv.org/abs/2610.32966> — [论文归档](../papers/vit_h_arxiv_2610_32966.md)
- **许可证：** MIT（仓库 LICENSE）
- **入库日期：** 2026-10-27
- **一句话说明：** ViT-H 论文的官方代码仓；目前公开仓库含 README、许可证与演示资产，训练、推理和部署代码尚未发布。
- **沉淀到 wiki：** 是 → [ViT-H 论文实体页](../../wiki/entities/paper-vit-h-video-to-humanoid.md)

## 仓库状态

仓库 README 自称是 ViT-H 论文的官方实现，并列出两项计划：离线 video-to-humanoid 控制代码目标于 2026-11-01 发布，humanoid video teleoperation 代码目标于 2026-12-01 发布。以 2026-10-27 入库时查看，仓库只有 assets、.gitignore、LICENSE 与 README.md，未见可执行训练、推理或部署入口。计划时间仅是 README 中的目标，不能当作已发布状态。

## 方法对照

| 组件 | 论文描述 | 当前仓库状态 |
|------|----------|--------------|
| 视频条件动作生成器 | 扩散模型从相机朝向、人体框/关键点和图像特征生成 G1 参考动作 | 代码计划发布 |
| tracker 与偏好后训练 | 执行候选动作并用跟踪反馈构造偏好对，DPO 更新生成器 | 代码未见公开 |
| 在线视频遥操作 | 自回归生成器按片段生成，驱动 G1 实时跟踪 | README 计划于 2026-12-01 发布 |

## 主要入口

- README / 发布计划：<https://github.com/mileret/ViT-H-release>
- MIT 许可证：<https://github.com/mileret/ViT-H-release/blob/main/LICENSE>
- 项目页与演示：<https://awfuact.github.io/vit-h/>

## Wiki 映射

- [ViT-H 论文实体页](../../wiki/entities/paper-vit-h-video-to-humanoid.md)
