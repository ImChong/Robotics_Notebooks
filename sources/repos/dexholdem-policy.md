# DexHoldem Policy（DexHoldem/Dexholdem-Policy）

> 来源归档（repo）

- **标题：** DexHoldem Policy — ShadowHand + UR 德州扑克模仿策略
- **类型：** repo / dexterous-manipulation / imitation-learning / benchmark
- **来源：** DexHoldem / 香港大学 IDS
- **链接：** <https://github.com/DexHoldem/Dexholdem-Policy>
- **论文：** [arXiv:2605.18727](https://arxiv.org/abs/2605.18727) — [`sources/papers/dexholdem_arxiv_2605_18727.md`](../papers/dexholdem_arxiv_2605_18727.md)
- **项目页：** <https://dexholdem.github.io/Dexholdem/> — [`sources/sites/dexholdem-github-io.md`](../sites/dexholdem-github-io.md)
- **数据集：** <https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot>
- **入库日期：** 2026-09-05
- **一句话说明：** 公开六条训练配方（DP / ACT / RDT）与 ZeroMQ 部署；π 系走 OpenPI 桥，不在这六条里。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-dexholdem.md`](../../wiki/entities/paper-dexholdem.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-09-05 复核） |
|----|-------------------|
| 训练 / 部署代码 | **已开源**：`scripts/train_*.sh`、`workflow/`、`deploy_policy.py`、`robot_client.py` |
| 权重 | 需自训；RDT_FT 另拉 `robotics-diffusion-transformer/rdt-1b` |
| 数据集 | [Winniechen2002/TexasPokerRobot](https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot)（CC BY 4.0，约 378 GB） |
| 许可证 | **未附 LICENSE**（GitHub `license: null`） |
| 环境 | Python 3.10、PyTorch cu121（README） |
| 公开配方 | DP(DINO)、DP_transformer_resnet、DP_unet、ACT、RDT_small、RDT_FT |

**结论：** **已开源可运行**。最短路径是子集下载 + `scripts/prepare.sh` + 任一条 `train_*.sh`。真机部署还要 ShadowHand + UR10e + 三路 RealSense。

## 仓库入口（对齐时序图）

| 路径 | 角色 |
|------|------|
| `workflow/download_data.py` | 从 HF 拉 TexasPokerRobot |
| `workflow/organize_data.py` | 每原语留 5 条 val，展开 `.npy` |
| `workflow/precompute_features.py` | DinoV2 CLS 或 SigLIP patch |
| `scripts/train_dp.sh` 等 | 六条公开训练入口 |
| `learning/dp/` `learning/act/` `learning/rdt/` | 模型实现 |
| `deploy_policy.py` | 默认 ZeroMQ `:13579` 出关节目标 |
| `robot_client.py` | 真机端采图 + 发 instruction ID |
| `docs/release_reproduction.md` | 端到端复现说明 |

## 关联资料

- 论文：[`sources/papers/dexholdem_arxiv_2605_18727.md`](../papers/dexholdem_arxiv_2605_18727.md)
- 项目页：[`sources/sites/dexholdem-github-io.md`](../sites/dexholdem-github-io.md)
- 技能仓：[`sources/repos/dexholdem-skills.md`](./dexholdem-skills.md)
- 数据集：[`sources/datasets/texaspokerrobot.md`](../datasets/texaspokerrobot.md)
- Wiki：[wiki/entities/paper-dexholdem.md](../../wiki/entities/paper-dexholdem.md)
