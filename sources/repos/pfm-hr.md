# gaoyukang33/PFM-HR

- **标题：** PFM-HR 官方代码仓
- **类型：** repo
- **URL：** <https://github.com/gaoyukang33/PFM-HR>
- **许可：** MIT（已有 `LICENSE`）
- **配套论文：** [arXiv:2608.03227](https://arxiv.org/abs/2608.03227) — [`sources/papers/pfm_hr_arxiv_2608_03227.md`](../papers/pfm_hr_arxiv_2608_03227.md)
- **项目页：** <https://gaoyukang33.github.io/PFM-HR.web/> — [`sources/sites/pfm-hr-web.md`](../sites/pfm-hr-web.md)
- **入库日期：** 2026-08-08

## 一句话说明

官方宣称的 PFM-HR 代码入口；截至入库日 README 仅写 **Coming Soon**，尚无可运行训练 / 推理 / 权重。

## 仓库状态（2026-08-08 核查）

| 项 | 内容 |
|----|------|
| default branch | `main` |
| tip 内容 | `LICENSE` + `README.md`（`# PFM-HR` / `Coming Soon ！！！`） |
| size | ≈2 KB |
| 训练入口 | **无** |
| 权重 / 数据 | **无** |
| homepage | 未设置（项目页见独立 GitHub Pages） |

## 与 wiki 的关系

- 实体页：[paper-pfm-hr](../../wiki/entities/paper-pfm-hr.md) — 源码运行时序图标注 **不适用**，待正式 release 后补 `sequenceDiagram`。
- 论文预期栈（非本仓现状）：无序姿态 Flow Matching 预训练 → 冻结先验 → ADD / BeyondMimic 跟踪中算 PGS 调制奖励。

## 跟进事项

- [ ] README 出现 install / train / eval 入口后补运行时序图
- [ ] 若发布 checkpoint 或 BONES-SEED 处理脚本，更新本页与实体页「开源」行
