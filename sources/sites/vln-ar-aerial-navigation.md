# VLN-AR | 官方项目页核验

> 官方项目主页来源归档（site）。

- **名称：** VLN-AR — A Hierarchical Onboard Vision-Language-Navigation Framework for Aerial Robots in Complex Environments
- **URL：** <https://wyz000.github.io/VLN-AR/>
- **论文：** <https://arxiv.org/abs/2611.23930v1>
- **机构：** 浙江大学工业控制技术全国重点实验室；Differential Robotics
- **检查日期：** 2026-11-22
- **核对范围：** 页面标题及作者、Overview、视频入口、Results（真机、长程、机载解码、鲁棒性、仿真表格）、Method、BibTeX，以及顶部和页尾的资源入口。
- **通读：** 已完成

## 官方页面能证实的内容

- 采用 3DGS 重建场景数据、三阶段训练、VLM + 记忆 Agent + action expert 的机载导航框架。
- 真机 **91/101（90.1%）**；长程 Agent 消融 **57.4%→88.0%**；Orin NX 30 W 上 2B–7B 每 token 解码 **7.1–9.4×**。
- 提供若干室内、室外、长程和鲁棒性飞行视频，以及方法介绍和 BibTeX DOI `10.1109/TRO.2026.3754505`。

## 资源开放核查

| 项目 | 核查结果 |
|------|----------|
| 论文 PDF / HTML | 已开放（arXiv v1） |
| 项目演示视频、实验表格 | 项目页公开 |
| 官方可运行源码 / README / 部署入口 | **未列链接**（截至 2026-11-22） |
| 模型权重、3DGS 训练数据、室内测试集下载 | **未列下载入口**（截至 2026-11-22）；正文提及室内测试集日后发布 |
| 源码运行时序图 | **不适用**：无法核验实际程序入口和模块调用 |

不能把基于论文 Fig. 1/3、Algorithm 1 整理的系统架构图称作源码复现。

## 对 wiki 的映射

- [论文全文核查归档](../papers/vln_ar_arxiv_2611_23930.md)
- [VLN-AR 论文实体](../../wiki/entities/paper-vln-ar-aerial-navigation.md)
