# Open-HumanWAM 官方项目页

> 来源归档（site）

- **标题：** Open-HumanWAM: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models
- **类型：** site
- **项目页：** <https://dagroup-pku.github.io/open-humanwam/>
- **arXiv：** <https://arxiv.org/abs/2611.06817>
- **论文全文：** <https://arxiv.org/html/2611.06817>
- **代码：** <https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale>
- **数据：** <https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K>
- **权重：** <https://huggingface.co/collections/DAGroup-PKU/open-humanwam>
- **入库日期：** 2026-11-06；2026-11-08 更新一手地址
- **一句话说明：** 50K 小时第一人称人类经验语料、HumanWAM-Vanilla 和 HumanWAM-Goal 的数据/训练/推理/评测配方。
- **沉淀到 wiki：** [Open-HumanWAM](../../wiki/entities/paper-open-humanwam.md)

## 核查结论

1. **官方项目页**采用 `open-humanwam/` 路径；旧的 `open-humanscale/` 是早期入口，知识库主链接应使用新路径。
2. **部分开源**（2026-11-08）：论文、部分源码、部分后训练/预训练权重已有官方入口；数据下载与某些权重/训练阶段仍处于上传或整理中。不能将“开放配方”误写成“所有实验素材齐备”。
3. **成果口径**：项目 README 报告 Vanilla Scene-OOD 49.1%→73.2%；Goal RoboTwin 最终 Clean/Random 92.70%/84.60%，zero-shot RoboTwin 11.48%→13.94%；详见[论文通读摘录](../papers/open_humanwam_arxiv_2611_06817.md)中的独立协议。
