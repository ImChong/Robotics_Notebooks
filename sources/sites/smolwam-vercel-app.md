# SmolWAM 项目页（smolwam.vercel.app）

> 来源归档

- **标题：** SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices
- **类型：** site（项目页）
- **URL：** <https://smolwam.vercel.app/>
- **论文：** <https://arxiv.org/abs/2610.13740>（归档见 [smolwam_arxiv_2610_13740.md](../papers/smolwam_arxiv_2610_13740.md)）
- **代码：** <https://github.com/iitimii/SmolWAM>（归档见 [smolwam.md](../repos/smolwam.md)）
- **权重：** <https://huggingface.co/timiiowolabi/SmolWAM>
- **入库日期 / 核查日期：** 2026-10-12
- **一句话说明：** ML Collective 项目页：“以教师 97% 成功率、11× 更低延迟”的边缘部署世界-动作模型。

## 页面核查结论

- 页面注明被 NeurIPS 2026 Workshop on Robot Learning with World Models 接收；链接论文 PDF、代码、HF 权重；称仓库内有安装、LIBERO 评测与训练文档，可 pip 从 GitHub 安装并从 HF 加载权重。
- 页面汇总数字与论文一致：LIBERO 95.5%（400 回合）vs 教师 98.2%，57 ms vs 624 ms，2.3 vs 42.6 TFLOP，2.36 vs 7.55 GiB，0.88B vs 1.96B，训练 3.6 GPU 小时。
- 页面列出负面结果（SmolWAM-LA 筛选 53.3%；世界头奖励加权微调塌到 5–16.7%）与作者自述限制（仅 LIBERO、长时域受剪枝影响、INT8 与端侧测量未完成）。
- 判定：**已开源**（代码 Apache-2.0 + 权重）。

## 对 wiki 的映射

- [SmolWAM 详情节点](../../wiki/entities/paper-smolwam.md)
