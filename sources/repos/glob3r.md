# Glob3R（aigc3d/Glob3R）

- **标题**: Glob3R: Global Structure-from-Motion with 3D Foundation Models
- **链接**: [https://github.com/aigc3d/Glob3R](https://github.com/aigc3d/Glob3R)
- **类型**: repo / placeholder
- **作者**: Deng, Junyuan, et al. (2026)；HKUST Spatial AI Lab · Alibaba Tongyi Lab · NJU · Fudan
- **项目页**: [https://junyuandeng.github.io/Glob3r/](https://junyuandeng.github.io/Glob3r/)
- **论文**: arXiv:2607.09225 — [`sources/papers/glob3r_arxiv_2607_09225.md`](../papers/glob3r_arxiv_2607_09225.md)
- **入库日期**: 2026-07-21
- **摘要**: 官方仓将托管 Glob3R 推理与评测；截至入库日 **仅公开 README**，声明将发布 Inference Code 与 Evaluation Script。

## 开源状态（截至 2026-07-21）

| 项 | 状态 |
|----|------|
| **仓库** | 已创建 `aigc3d/Glob3R`（项目页 Code 按钮指向此处） |
| **可运行入口** | **无** — 树内仅 `README.md` |
| **TODO（README）** | `[ ] Inference Code Release`；`[ ] Evaluation Script` |
| **权重 / 数据** | README **未** 列 HF/ModelScope 链接 |
| **结论** | **部分开源（占位仓）/ 推理待发布** — 有官方 URL，尚无可辨识训练/推理脚本 |

## README 要点

1. **方法定位**: 可扩展全局 3D 重建；把前馈稠密预测转为可靠多视图 tracks，再经 motion averaging + BA 联合精炼位姿与几何。
2. **致谢 / 设计灵感**: Sail-Recon（PSNR 评测）、Pi3、RoMaV2、Instant-SfM、GLOMAP。
3. **引用**: 仓库 BibTeX 仍写 Anonymous Authors；以 arXiv / 项目页作者列表为准。

## 为什么值得保留

- 锁定复现入口与开放边界：读者勿把「项目页有 Code」等同于「今日可跑通」。
- 后续 Inference 放出后，可在此档补 CLI / 权重路径，并回填 wiki「源码运行时序图」。

## 对 wiki 的映射

- [wiki/entities/paper-glob3r.md](../../wiki/entities/paper-glob3r.md)
- [wiki/methods/lingbot-map.md](../../wiki/methods/lingbot-map.md) — 论文对标的流式前馈重建基线
