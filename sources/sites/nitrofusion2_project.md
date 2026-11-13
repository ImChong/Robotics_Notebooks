# NitroFusion2 — 官方项目主页核对

- **类型：** site / official project page
- **主页：** https://chendaryen.github.io/NitroFusion2.github.io/
- **论文：** https://arxiv.org/abs/2611.05648
- **代码：** https://github.com/ChenDarYen/NitroFusion2
- **权重：** https://huggingface.co/ChenDY/NitroFusion2
- **在线 Demo：** https://huggingface.co/spaces/ChenDY/NitroFusion2
- **入库日期：** 2026-11-13
- **核对范围：** 通读项目主页的头部资源入口、Architecture 方法图对应说明、Abstract、Citation 和页尾；逐项核对 GitHub、arXiv、Weights、Demo 四类资源可解析的目标。
- **通读：** 已完成
- **开放状态：** 官方项目页明确列出 GitHub、Weights 和 Demo。GitHub 已发布推理实现及 ComfyUI 工作流；未确认已发布论文完整训练代码。

## 一手信息

项目主张：MMDiT 文本 token 在 joint attention 中已吸收图像相关特征，因此可在文本流追加判别器；通过 typography/object/style 等属性 cue，重复利用一个冻结判别骨干引导不同视觉维度。它保持原有图像流判别监督并为文本流添加专家头，目标是让 1–2 步生成接近多步质量。

主站图片为展示结果，不应当作独立基准实验；具体数值需引用 arXiv §4 表格。

## 对 wiki 的映射

- [NitroFusion2 唯一项目实体](../../wiki/entities/paper-nitrofusion2.md)
- [论文全文通读归档](../papers/nitrofusion2_arxiv_2611_05648.md)
- [官方仓库与源码路径](../repos/nitrofusion2.md)
