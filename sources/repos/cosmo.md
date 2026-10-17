# Yangming911/CoSMO

> 来源归档

- **标题：** CoSMO 官方仿真代码
- **类型：** repo
- **代码：** <https://github.com/Yangming911/CoSMO>
- **论文：** [arXiv:2610.18321](https://arxiv.org/abs/2610.18321) — 归档见 [`sources/papers/cosmo_arxiv_2610_18321.md`](../papers/cosmo_arxiv_2610_18321.md)
- **项目页：** <https://yangming911.github.io/cosmo-page/> — 归档见 [`sources/sites/cosmo-project.md`](../sites/cosmo-project.md)
- **入库日期：** 2026-10-17（核查日同日）
- **许可：** MIT
- **一句话说明：** Habitat-Sim 上的 ObjectNav 评测：四份 YAML 配置对应 OneMap 基线、GCLIP argmax、CoSMO（真值专家）与 CoSMO（Grounding DINO 专家）。

## 环境与入口（README 摘要，未本地运行）

- 依赖：Python 3.8、PyTorch 2.4.1、CUDA 12.1、Habitat-Sim 0.2.2、CUDA GPU；YOLOv7 需另行克隆。
- 不含数据与权重：需自备 MP3D ObjectNav episodes、场景资产，以及 CLIP、YOLOv7、MobileSAM、（可选）Grounding DINO 检查点。
- 运行：在 `cosmo_code/` 下用 `eval_habitat.py` 加配置文件逐条件启动。
- 输出：逐 episode 状态、轨迹、相似度图、碰撞原因与校准历史，以及 SR / SPL / 运行时间汇总。
- 无 releases；引用处为占位。

## 与本仓库知识的关系

[CoSMO 实体页](../../wiki/entities/paper-cosmo.md)
