# hustvl.github.io/DreamWAM（DreamWAM 项目页）

- **标题：** DreamWAM — Beyond RGB Future Prediction for World Action Models
- **类型：** site / project-page
- **URL：** <https://hustvl.github.io/DreamWAM/>
- **配套论文：** [DreamWAM（arXiv:2608.04996）](https://arxiv.org/abs/2608.04996) — 归档见 [`sources/papers/dreamwam_arxiv_2608_04996.md`](../papers/dreamwam_arxiv_2608_04996.md)
- **代码：** <https://github.com/hustvl/DreamWAM> — [`sources/repos/dreamwam.md`](../repos/dreamwam.md)
- **权重：** <https://huggingface.co/hustvl/DreamWAM>
- **入库日期：** 2026-08-07

## 一句话摘要

华中科技大学 hustvl 等官方站点：展示 DreamWAM 用 appearance / motion / geometry / semantics 结构化未来预测训练 Joint WAM，并在部署时保持 RGB-only 推理；给出 LIBERO、LIBERO-Plus 与真机扰动对照表及 rollout。

## 公开信息要点（截至入库日）

- **页首指标：** LIBERO Average **98.90%**；LIBERO-Plus Average **75.47%**；真机扰动平均 **74.40%**。
- **入口：** Paper / Code / Models 链接齐全（步骤 2.5 → **已开源**）。
- **方法叙事：** VideoDiT 联合去噪 RGB+Flow；DINO/Depth 门控残差；Video–Action shared attention；推理关闭 beyond-RGB。
- **仿真表：** 与 Fast-WAM-Joint 同骨干 / 同数据 / 同协议的 matched 对照。
- **真机：** 四项桌面任务 + 光照 / 背景 / distractor 扰动；含 Fast-WAM-Joint vs DreamWAM 视频对照。

## 为何值得保留

- **步骤 2.5 开源核查主入口：** Code + HF Models 可点。
- **数字与 PDF 摘要互证：** LIBERO-Plus 与真机扰动增益是选型主读点。
- **WAM「预测什么」议题：** 相对「更强视频骨干」路线，项目页把贡献钉在 **未来表征形态**。

## 关联资料

- 论文归档：[`sources/papers/dreamwam_arxiv_2608_04996.md`](../papers/dreamwam_arxiv_2608_04996.md)
- 代码归档：[`sources/repos/dreamwam.md`](../repos/dreamwam.md)
- Wiki 实体：[wiki/entities/paper-dreamwam.md](../../wiki/entities/paper-dreamwam.md)
