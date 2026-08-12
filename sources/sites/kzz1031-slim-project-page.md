# SLIM-0.5B 项目页

> 来源归档

- **标题：** SLIM-0.5B — Learning Action-Grounded Predictive Latents for Robot Manipulation
- **类型：** site / project-page
- **URL：** <https://kzz1031.github.io/slim-project-page/>
- **论文：** <https://arxiv.org/abs/2608.09771> — [`sources/papers/slim_05b_arxiv_2608_09771.md`](../papers/slim_05b_arxiv_2608_09771.md)
- **代码：** <https://github.com/kzz1031/SLIM> — [`sources/repos/slim.md`](../repos/slim.md)
- **权重：** <https://huggingface.co/kzzwang/SLIM-LIBERO> · <https://huggingface.co/kzzwang/SLIM-CALVIN>
- **机构：** Fudan / BAAI / Tsinghua / RUC
- **入库日期：** 2026-08-12
- **一句话说明：** SLIM 官方项目站：掩码轨迹学习叙事、LIBERO / LIBERO-Plus / CALVIN / 真机数字与消融。

## 开源核查（步骤 2.5，截至 2026-08-12）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **是**（经论文/站内资源指向 GitHub `kzz1031/SLIM`） |
| 权重 / 数据 | **是** — HF `kzzwang/SLIM-LIBERO`、`kzzwang/SLIM-CALVIN`（完整 release 目录） |
| 综合判定 | **已开源**（训练 + 评测 + 权重） |

## 页面要点

- 0.47B MoT + masked trajectory（IDM/FDM）+ flow-matching。
- LIBERO **97.5%**；LIBERO-Plus **77.45%**；CALVIN avg len **4.556**。
- 真机 avg progress **67.8**；latency **77.3 ms**；GPU mem **2.01 GiB**。
- 消融：Stage-1、IDM:FDM=0.125:1、EMA 对 OOD/长程关键。

## 关联资料

- 论文：[`sources/papers/slim_05b_arxiv_2608_09771.md`](../papers/slim_05b_arxiv_2608_09771.md)
- 仓库：[`sources/repos/slim.md`](../repos/slim.md)
- Wiki：[`wiki/entities/paper-slim-05b.md`](../../wiki/entities/paper-slim-05b.md)
