# zhangdana483/real_bi_dex_grasp

- **标题：** 真机双臂灵巧抓取官方实现
- **类型：** repo
- **URL：** <https://github.com/zhangdana483/real_bi_dex_grasp>
- **许可：** 代码 Apache-2.0；数据集仅非商用研究
- **配套论文：** [arXiv:2608.10383](https://arxiv.org/abs/2608.10383) — [`sources/papers/real_bi_dex_grasp_arxiv_2608_10383.md`](../papers/real_bi_dex_grasp_arxiv_2608_10383.md)
- **入库日期：** 2026-08-18

## 一句话说明

DDPM 关节级双臂抓取：训练/推理脚本 + 遥操作采集 + 数据样例；全集约 40GB 网盘。

## 仓库状态（2026-08-18 核查）

| 项 | 内容 |
|----|------|
| 训练 | `ddpm_model/train_ddpm.py` |
| 推理 | `ddpm_model/infer_ddpm.py` |
| 采集 | `avp_teleoperate/`（基于 Unitree 开源） |
| 数据 | `dataset/` 样例；百度网盘链接见 README |

最短复现：按 `description.txt` 理解字段 → 用样例跑 `infer_ddpm.py`；完整训练需下载全集。

## 与 wiki 的关系

- 实体页：[paper-real-bi-dex-grasp](../../wiki/entities/paper-real-bi-dex-grasp.md) — 含源码运行时序图。
