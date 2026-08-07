# DreamWAM（hustvl/DreamWAM）

> 来源归档（repo）

- **标题：** DreamWAM — Beyond RGB Future Prediction for World Action Models
- **类型：** repo / world-action-models / joint-wam / libero
- **来源：** hustvl（华中科技大学）
- **链接：** <https://github.com/hustvl/DreamWAM>
- **论文：** [arXiv:2608.04996](https://arxiv.org/abs/2608.04996) — 归档见 [`sources/papers/dreamwam_arxiv_2608_04996.md`](../papers/dreamwam_arxiv_2608_04996.md)
- **项目页：** <https://hustvl.github.io/DreamWAM/> — [`sources/sites/hustvl-dreamwam-github-io.md`](../sites/hustvl-dreamwam-github-io.md)
- **权重：** <https://huggingface.co/hustvl/DreamWAM>
- **Stars：** ~17（2026-08-07）
- **入库日期：** 2026-08-07
- **一句话说明：** DreamWAM 官方训练 / 评测栈：基于 FastWAM 的 VideoDiT–ActionDiT，附加 flow/depth/DINO 预处理与 LIBERO / LIBERO-Plus 评测入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-dreamwam.md`](../../wiki/entities/paper-dreamwam.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-08-07） |
|----|-------------------|
| 训练 / 评测代码 | **已开源**（`scripts/train.py`、`eval_libero.py`、`eval_libero_plus.py`） |
| 预处理缓存 | `scripts/precompute_cache.py`（RGB+flow latent + DINO/Depth 目标） |
| 权重 | HF `dreamwam_joint.pt` / `dreamwam_uncond.pt`（MIT） |
| 依赖 | Wan2.2 TI2V-5B、RAFT、DINOv2、Depth-Anything-3、LIBERO / LIBERO-Plus |
| 许可证 | HF **MIT**；GitHub 根目录未声明 SPDX（截至核查日） |

**结论：** **已开源可运行实现**；完整复现需下载 Wan/RAFT/DINO/DA3 与 FastWAM 风格 LIBERO 数据。

---

## README 宣称的技术栈 / 入口

| 组件 | 路径 / 命令 |
|------|-------------|
| 环境 | Python 3.10 + CUDA 12.x；`pip install -e .` + DA3 / LIBERO |
| ActionDiT 初始化 | `scripts/prepare_action_dit.py --config configs/dreamwam_joint.yaml` |
| 预处理 | `scripts/precompute_cache.py` → `cache/libero_2cam224` |
| 训练 | `accelerate launch … scripts/train.py --config configs/dreamwam_{uncond,joint}.yaml` |
| LIBERO 评测 | `scripts/eval_libero.py` |
| LIBERO-Plus | `scripts/eval_libero_plus.py`（10,030-task 协议） |
| 配置 | `configs/dreamwam_joint.yaml` / `dreamwam_uncond.yaml` |

## 关联资料

- 论文归档：[`sources/papers/dreamwam_arxiv_2608_04996.md`](../papers/dreamwam_arxiv_2608_04996.md)
- 项目页：[`sources/sites/hustvl-dreamwam-github-io.md`](../sites/hustvl-dreamwam-github-io.md)
- Wiki 实体：[wiki/entities/paper-dreamwam.md](../../wiki/entities/paper-dreamwam.md)
