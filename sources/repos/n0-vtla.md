# N0-VTLA（neoteai/N0-VTLA）

> 来源归档

- **标题：** N₀-VTLA — Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens
- **类型：** repo
- **组织：** NeoteAI × Fudan TEAI
- **代码：** <https://github.com/neoteai/N0-VTLA>
- **项目页：** <https://research.neoteai.com/n0-vtla/>
- **技术报告：** <https://research.neoteai.com/assets/n0-vtla-report.pdf>
- **许可：** README badge **CC-BY-NC-SA-4.0**
- **入库日期：** 2026-07-26
- **一句话说明：** 𝒩₀-VTLA 官方仓；Roadmap 计划 **2026-07-31** 前发布模型代码、预训练/后训练权重与配方；截至入库日仅 README + diagrams。
- **沉淀到 wiki：** [𝒩₀-VTLA（论文实体）](../../wiki/entities/paper-n0-vtla.md)

## 开源状态（步骤 2.5）

| 项 | 状态 |
|----|------|
| 模型 / 推理 / 后训练代码 | **待发布**（By July 31, 2026） |
| 预训练与后训练权重 | **待发布** |
| 可运行入口 | **无** |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **预测触觉 latent** 条件 flow-matching 动作专家，而非把触觉当额外相机 |
| [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md) | 接触差分 token + 未来接触预测，对应「接触前视觉 / 接触期触觉」 |
| [TACO](../../wiki/entities/paper-taco-tactile-wm-vla-posttrain.md) | ALTER 离线 advantage 与失败/纠正数据利用可对照 |
| [𝒩₀-Foundation](../../wiki/entities/paper-n0-foundation.md) | 预训练数据与触觉表征底座 |
