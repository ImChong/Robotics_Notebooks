# OmniCompass 数据集来源

> 来源归档

- **类型：** dataset
- **数据集：** <https://huggingface.co/datasets/VCLab-PolyU/OmniCompass>
- **ModelScope 镜像：** <https://modelscope.cn/datasets/liuxuyang16/OmniCompass>
- **论文：** <https://arxiv.org/abs/2610.18829>
- **项目页：** <https://omnicompass.netlify.app/>
- **代码与评测器：** <https://github.com/PolyU-VCLab/OmniCompass>
- **许可证：** CC BY-NC-SA 4.0（Hugging Face 数据卡）
- **入库日期：** 2026-10-25

## 已核实信息

- 论文及官方 README 报告 OmniCompass-VL 有 820 个视频、2,460 个实例；OmniCompass-AV 有原生音频的 240 个视频、720 个实例。
- Hugging Face 页面当前可见 OmniCompass-VL 的 test split，共 2,460 行；字段包括视频 ID、任务、目标描述、标注答案、视频路径、音频标志和来源视频等。
- 任务字段为 TG（时序定位）、SG（空间定位）、QA（问答）；卡片标注 English、Parquet、CC BY-NC-SA 4.0。
- HF viewer 当前展示 VL split；不能只凭论文中的 AV 子集描述推断 AV 文件已在该卡完整发布。使用前检查 Files 标签与官方镜像。
- 公开数据许可证带有非商业与相同方式共享条件；上游视频来源还可能有各自权利和平台条款。

## Robotics_Notebooks 映射

- 项目实体：[OmniCompass](../../wiki/entities/paper-omnicompass.md)
- 论文：[omnicompass_arxiv_2610_18829](../papers/omnicompass_arxiv_2610_18829.md)
- 项目页：[omnicompass-netlify-app](../sites/omnicompass-netlify-app.md)
- 仓库：[polyu-vclab-omnicompass](../repos/polyu-vclab-omnicompass.md)