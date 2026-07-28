# CondMDI 项目页

> 来源归档（ingest）

- **标题：** Flexible Motion In-betweening with Diffusion Models
- **类型：** site / project-page
- **官方入口：** <https://setarehc.github.io/CondMDI/>
- **论文：** <https://arxiv.org/abs/2405.11126>
- **代码：** <https://github.com/setarehc/diffusion-motion-inbetweening>
- **入库日期：** 2026-07-28
- **一句话说明：** CondMDI（SIGGRAPH 2024）项目页，展示稀疏/密集/部分关键帧、根轨迹与文本联合条件的动作补全结果。
- **开源状态（2026-07-28 核查）：** **已开源** — 官方 PyTorch 仓库含训练、推理、评测入口与三个 HumanML3D 预训练权重；仓库 `LICENSE` 为 MIT。

## 页面公开信息

| 资源 | URL |
|------|-----|
| 项目首页 | <https://setarehc.github.io/CondMDI/> |
| arXiv | <https://arxiv.org/abs/2405.11126> |
| Code | <https://github.com/setarehc/diffusion-motion-inbetweening> |

项目页对比纯插补、重建引导、GMD 与 OmniControl，并明确 CondMDI 训练时随机抽取帧与关节掩码，以支持推理时任意关键帧布局。

## 对 wiki 的映射

- [`wiki/entities/paper-notebook-flexible-motion-in-betweening-with-diffusion-mod.md`](../../wiki/entities/paper-notebook-flexible-motion-in-betweening-with-diffusion-mod.md)
- [`sources/repos/condmdi.md`](../repos/condmdi.md)
- [`sources/papers/humanoid_pnb_flexible-motion-in-betweening-with-diffusion-mod.md`](../papers/humanoid_pnb_flexible-motion-in-betweening-with-diffusion-mod.md)
