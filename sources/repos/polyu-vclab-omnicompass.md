# OmniCompass 官方代码仓库

> 来源归档

- **标题：** OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding
- **类型：** repo
- **链接：** <https://github.com/PolyU-VCLab/OmniCompass>
- **论文：** <https://arxiv.org/abs/2610.18829>
- **项目页：** <https://omnicompass.netlify.app/>
- **数据集：** <https://huggingface.co/datasets/VCLab-PolyU/OmniCompass>
- **入库日期：** 2026-10-25
- **一句话说明：** 提供 OmniCompass-VL / AV 评测工具，可通过统一 run.sh 对 API 或自托管模型运行 TG、SG、QA。

## 开源核查（2026-10-25）

| 资源 | 状态 | 说明 |
|------|------|------|
| 评测代码 | **已开源** | 仓库公开，含 run.sh、code/、依赖文件及测试 |
| 数据 | **已发布 / 子集待核验** | README 指向 HF 与 ModelScope；HF 页面显示 VL 的 2,460 行测试 split，AV 实际可下载范围需检查文件列表 |
| 项目页 | **已公布** | README 将 Netlify 项目页列为官方入口；本次抓取工具无法直接打开该站点 |
| 软件许可证 | **仓库根目录未见 LICENSE** | 不推断代码可按特定开源许可证再分发 |

## README 核查要点

- run.sh 是 API 模型与自托管模型共用的评测入口。
- 可配置 VL / AV 轨道、TG / SG / QA 任务、text / text+image 设定、帧采样、音频、并发、缓存和输出路径。
- runner 抽取并缓存视频帧；AV 音频评测会抽取音轨，支持断点续跑。输出包括 predictions.jsonl、scores.json 与 scores.csv。
- README 提供 omnicompass-eval agent skill；评测者需准备数据和兼容模型端点。
- 数据集 HF 卡标注 CC BY-NC-SA 4.0；数据使用和再分发须遵循相应条件。

## 关联来源

- 论文：[OmniCompass](../papers/omnicompass_arxiv_2610_18829.md)
- 项目页：[OmniCompass](../sites/omnicompass-netlify-app.md)
- 数据集：[OmniCompass Dataset](../datasets/vclab-polyu-omnicompass.md)
- wiki 实体：[paper-omnicompass](../../wiki/entities/paper-omnicompass.md)