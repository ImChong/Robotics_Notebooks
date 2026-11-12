# THAW-VLA 项目页（thaw-vla.trung-dt.com）

> 来源归档（site）

- **标题：** THAW-VLA: Think Like a World Model, Act Like a VLA
- **类型：** project-page
- **URL：** <https://thaw-vla.trung-dt.com/>
- **论文：** [arXiv:2609.24682](../papers/thaw_vla_arxiv_2609_24682.md)
- **机构：** University of Wisconsin–Madison、University of Illinois Urbana-Champaign
- **入库日期：** 2026-09-23
- **复核日期：** 2026-11-12
- **核对范围：** 已阅读项目页方法、表格、消融、真机说明和源码 / 权重链接，并核对模型页面的公开文件列表。
- **通读：** 已完成
- **代码：** **已开源** — <https://github.com/trungdt880/THAW-VLA>；归档 [`sources/repos/thaw-vla.md`](../repos/thaw-vla.md)
- **权重：** <https://huggingface.co/collections/termanteus/thaw-vla>（两种 0.8B checkpoint 的文件列表可公开访问；作者 README 的 private 提示尚未更新）
- **一句话说明：** 0.8B QwenGR00T 经 Cosmos3-Nano 特征蒸馏后在 LIBERO 97.9%、RoboCasa-GR1 50.5%、真机 66.7% 均值；部署与 undistilled 同图（32 ms / 1.86 GB RTX 5090）。

## 核查结论（步骤 2.5）

- **已公开：** 方法动画、LIBERO/GR1/真机结果表、teacher/backbone/layer 消融、真机 rollout 视频、BibTeX
- **已开源：** GitHub 含 setup / precompute / train / eval / websocket deployment；MIT license
- **权重公开情况：** Hugging Face 页面可匿名看到 LIBERO 和 GR1 模型的 `final_model.pt`；本次未完整下载权重，也未做推理验证。Cosmos3-Nano teacher checkpoint 需用户自备
- **Footer / 页内链接：** arXiv、GitHub（README badge）、Hugging Face collection

## 页面要点摘录

- **学生：** Qwen3-VL backbone + flow-matching action expert；alignment 后 projector 移除
- **Teacher 管线：** Cosmos3-Nano 冻结；layer-24 image tokens；每视角 mean-pool
- **真机：** AgileX Nero 单臂（fruit/egg pick-place）；TRIP-Bag 双臂 fruit handover
- **失败模式（真机）：** 多为 **placement 末段厘米级误差**，非识别错误
