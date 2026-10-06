# zju3dv/INTACT-JEPA

> 来源归档（ingest 配套仓库 · 规范仓）

- **URL：** <https://github.com/zju3dv/INTACT-JEPA>
- **对应论文：** [arXiv:2607.26056](https://arxiv.org/abs/2607.26056)
- **项目页：** <https://zju3dv.github.io/INTACT-JEPA/>
- **RoboParty 组织镜像：** <https://github.com/Roboparty/INTACT-JEPA> → 见 [roboparty-intact-jepa.md](roboparty-intact-jepa.md)
- **Lab：** <https://lab.roboparty.com/>
- **许可：** MIT
- **入库日期：** 2026-07-30
- **最后更新：** 2026-10-06
- **一句话说明：** INTACT **规范仓**：已公开训练、评测、复现契约及 checkpoint 下载入口；组织 fork 仍为旧预览，不代表上游当前状态。
- **代码：** <https://github.com/zju3dv/INTACT-JEPA>（`scripts/train.sh`、`scripts/train_multitask.sh`、`scripts/eval.sh`）
- **权重：** <https://huggingface.co/INTACT-JEPA/INTACT>（README 另列统一模型与消融入口，具体 revision 按 manifest）

## 仓库要点（2026-07 快照）

| 路径 / 徽章 | 状态 |
|-------------|------|
| `README.md` / `README_CN.md` | 方法 TL;DR、结果摘要；作者单位含 **RoboParty Lab** |
| `docs/METHOD.md` | 方法笔记 |
| `docs/RESULTS.md` | 审计结果 |
| `docs/REPRODUCIBILITY.md` | 复现契约 |
| `docs/RELEASE.md` | Stage 0–3 发布计划 |
| Code / Models badge | **Coming Soon** |

## 开源边界

**当前核查（2026-10-06）：** 先打开[项目页](https://zju3dv.github.io/INTACT-JEPA/)，其 Code 链到本规范仓；上游 README 已给出安装、数据校验、smoke、训练、评测、权重下载和校验入口。代码与权重已有公开入口；数据复用官方 LeWM，须独立获取和转换，训练脚本不隐式下载。RoboParty fork 仍是研究预览，下面的「待发」记录仅代表 2026-07 快照。

| 当前运行入口 | 职责 |
| --- | --- |
| `scripts/install.sh`、`scripts/verify_install.py` | 锁定环境与 CUDA 检查 |
| `scripts/verify_data.py` | 检查本地 LeWM 数据布局 |
| `scripts/train.sh`、`scripts/train_multitask.sh` | 单任务 / 共享编码器训练，支持 smoke |
| `scripts/eval.sh` | Direct、CEM 与 Guarded-A 评测 |
| `scripts/download_paper_checkpoints.sh`、`paper_runtime/` | 权重下载及论文 checkpoint 兼容运行时 |

本次仅核查公开入口，未实际训练或验证任务成功率；当前实现、论文 checkpoint 与旧预览中的数字按版本分别读取。

### 历史边界（2026-07-30）

- **已有：** 术语、图表、headline 数字、复现/署名约定、MIT LICENSE。
- **待发：** 训练与评测源码、配置与 checkpoint manifest、代表性权重（Stage 2+）。
- 官方另提及评测生态 LeWM / CLEAR-LeWM；勿把「仓已存在」误读为「可本地训通」。
- **镜像：** [Roboparty/INTACT-JEPA](https://github.com/Roboparty/INTACT-JEPA) 为上游 fork，内容同构；导航可用，**不以镜像替代规范仓版本锚定**。

## 对 wiki 的映射

- [INTACT 论文实体](../../wiki/entities/paper-intact.md)
- [论文归档](../papers/intact_arxiv_2607_26056.md)
- [RoboParty 镜像归档](roboparty-intact-jepa.md)
