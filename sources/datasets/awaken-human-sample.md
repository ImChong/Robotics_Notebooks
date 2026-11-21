# Awaken-Human：公开数据样例与完整语料的区分

> 一手 Hugging Face 数据卡归档；仅描述当前实际分发内容，不镜像数据。

- **类型：** dataset
- **入口：** <https://huggingface.co/datasets/EngineAI-Embodied/Awaken-Human>
- **论文：** <https://arxiv.org/abs/2611.26420>
- **项目：** <https://engineai-robotics.github.io/Awaken-HI/>
- **入库日期：** 2026-11-21
- **对应实体：** [Awaken-HI](../../wiki/entities/paper-awaken-hi.md)
- **对应论文：** [精读归档](../papers/awaken_hi_arxiv_2611_26420.md)

## 当前可下载资源

- 官方卡明确：“This repository currently distributes one complete sample recording”。sample 配置 / sample split 对应同一条录制。
- 录制 33.8 秒、1,014 帧、30 fps、512×384 H264，包含 8 段时序动作标注；原始文件包括去畸变左眼视频、时间戳全身/手部追踪、相机内外参与动作标签 JSON，并有 Parquet 索引。
- 许可 **engineai-academic-noncommercial-no-redistribution-1.0**：仅限非商业学术研究，不可商业使用、转售或再分发（含重传、镜像、修改后分发）；英语 LICENSE 为准。
- 论文中的 50,000 h / 17.02M episodes / 5.41B 姿态帧为**作者构建语料的报告规模**，**不是当前 HF 公共仓库的实际下载体量**。研究复现需等待全量语料后续放出并检查许可。
