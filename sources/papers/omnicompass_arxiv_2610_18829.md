# OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding（arXiv:2610.18829）

> 来源归档（ingest）

- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2610.18829>（v1，2026-10-13）
- **PDF：** <https://arxiv.org/pdf/2610.18829>
- **作者：** Xuyang Liu、Shihao Wang、Xinyu Wei、Minghan Li、Zeqing Wang、Zuhao Yang、Zichen Wen、Lei Zhang
- **机构：** 香港理工大学、OPPO 研究院、科罗拉多矿业学院、南洋理工大学、上海交通大学
- **项目页：** <https://omnicompass.netlify.app/>
- **代码：** <https://github.com/PolyU-VCLab/OmniCompass>
- **数据集：** <https://huggingface.co/datasets/VCLab-PolyU/OmniCompass>
- **入库日期：** 2026-10-25
- **一句话说明：** 用配对的文本与文本+图像目标描述，评测视频模型的时序定位、空间定位、问答及音视频理解能力。

## 核心论文摘录

### 1) 多维视频 grounding 基准设计

- **链接：** <https://arxiv.org/abs/2610.18829>
- **摘录要点：** 论文提出 OmniCompass-VL 与 OmniCompass-AV 两条轨道，覆盖 temporal grounding（何时）、spatial grounding（何处）和 question answering（发生了什么）；同一目标分别用文本、文本+图像描述，比较目标规格变化下的表现与一致性。
- **对 wiki 的映射：** [OmniCompass 论文实体页](../../wiki/entities/paper-omnicompass.md)、[具身评测基准选型闭环](../../wiki/overview/hub-embodied-eval-benchmark.md)

### 2) 视频规模与人工标注

- **链接：** <https://arxiv.org/abs/2610.18829>
- **摘录要点：** OmniCompass-VL 包含 820 个视频、2,460 个实例；含原生音频的 OmniCompass-AV 子集包含 240 个视频、720 个实例。作者称数据经过多轮人工质量核验，累计约 1,800 人时。
- **对 wiki 的映射：** [OmniCompass 论文实体页](../../wiki/entities/paper-omnicompass.md)、[Hugging Face 数据集归档](../datasets/vclab-polyu-omnicompass.md)

### 3) 评测范围与主要发现

- **链接：** <https://arxiv.org/abs/2610.18829>
- **摘录要点：** 论文评估 50 个多模态大模型，报告任务表现与跨目标规格一致性。摘要指出，开源模型整体落后于专有模型，音视频理解差距尤其明显；精准时空定位对领先模型仍有挑战，配对规格下也会出现预测不一致。
- **对 wiki 的映射：** [OmniCompass 论文实体页](../../wiki/entities/paper-omnicompass.md)、[具身评测基准选型闭环](../../wiki/overview/hub-embodied-eval-benchmark.md)

## 来源与身份说明

论文、项目页、代码与数据集属于同一 OmniCompass 基准，统一编译至 [paper-omnicompass](../../wiki/entities/paper-omnicompass.md)。项目页核查与代码可运行状态见[项目页归档](../sites/omnicompass-netlify-app.md)及[官方仓库归档](../repos/polyu-vclab-omnicompass.md)。