# DiffWAM 官方项目页与开放状态核查

> 来源归档；核查日期：2026-10-12

- **项目页：** <https://zzmmzzm.github.io/diffwam.github.io/>
- **论文：** <https://arxiv.org/abs/2609.39763> — [论文归档](../papers/diffwam_arxiv_2609_39763.md)
- **NavGen 配套项目：** <https://xinjiu612.github.io/NavGen/>
- **FlyMirage 配套项目：** <https://lij1500.github.io/flymirage_website/>
- **已核查配套数据：** <https://huggingface.co/datasets/Li-200/flymirage> — card 标注首批 50,000 轨迹、MIT；不是 DiffWAM 权重或完整训练集。
- **类型：** site / UAV world-action model

## 页面与开放核查

页面公开展示方法、轨迹演示、实验与板卡数据。未发现 DiffWAM 专属模型代码、检查点或 benchmark 下载按钮；配套数据项目链接不等于 DiffWAM 实现。FlyMirage 页面确有 Hugging Face 数据入口，论文也称配套项目已发布初始批次，但不能据此断言超过 510K 训练样本及 DiffWAM-1000 全部开放。

正文有标准/Flash 推理预算和部分 ADE/FDE 数值混用，页末还有历史 development-set 边界。收录以 arXiv v1 具体章节和表格为准；官方页本身不作为代码许可证或实时安全保证。

## 对 wiki 的映射

- [DiffWAM 统一实体](../../wiki/entities/paper-diffwam.md)
