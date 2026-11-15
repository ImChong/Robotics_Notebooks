# AesFormer: Transform Everyday Photos into Beautiful Memories（ICML 2026）

> 论文来源归档（依据官方 arXiv v1 全文与官方仓库整理，不是论文全文转载）。

- **标题：** AesFormer: Transform Everyday Photos into Beautiful Memories / AesFormer：将日常照片重建为美好回忆
- **作者：** Tianxiang Du、Hulingxiao He、Yuxin Peng
- **机构：** 北京大学王选计算机研究所（Wangxuan Institute of Computer Technology, Peking University）
- **类型：** paper / aesthetic-photo-reconstruction / image-editing / planning / GRPO
- **论文：** <https://arxiv.org/abs/2605.22126>；[v1 HTML](https://arxiv.org/html/2605.22126v1)；[PDF](https://arxiv.org/pdf/2605.22126)
- **版本：** arXiv:2605.22126v1（2026-05-21）；论文及仓库注明 ICML 2026 录用。
- **源码：** <https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026>；详见 [源码归档](../repos/aesformer-icml2026.md)
- **入库日期：** 2026-11-15
- **核对范围：** arXiv v1 官方 HTML 与 PDF：摘要、§1–§6（动机、VCMP/AesRecon、AesThinker SFT/GRPO-A、AesEditor、实验 Table 1/2、局限与结论），以及官方 README/训练评测入口；PDF 共 11 页，文末为参考文献。
- **通读：** 已完成
- **一句话说明：** 定义美学照片重建 APR，用「美学规划 AesThinker → 动作条件编辑 AesEditor」两阶段改善构图/视角/姿态等结构，同时约束人物身份与场景语义不变。
- **对 wiki 的映射：** [AesFormer 论文实体](../../wiki/entities/paper-aesformer.md)；交叉关联 [GRPO](../../wiki/methods/grpo.md)。

## 核心摘录 1：APR 任务（§1–2）

传统修图主要改变色调、曝光、皮肤等外观；APR 允许调整**构图、机位、人物位置与姿态**等拍摄时的结构错误，但须保留人物身份及场景语义。此处的“动作”是**照片编辑指令**，并非机器人执行器控制动作。

## 核心摘录 2：VCMP → AesRecon（§3）

- 关键词检索 **5,700** 个候选视频，筛选至 **2,144** 段摄影教学视频；按拍摄事件切分并以 **2 fps** 采样。
- 使用 Qwen2.5-VL-72B 检出示范中的优质照片帧；以事件首帧作初始差照片构成粗配对。
- 质量/美学与转场过滤；Qwen-Image-Edit-2511 去除字幕/相机 UI，GPT-4o 审核清理；VLM 核对同一主体、场景和拍摄事件，拒绝新物体或身份变化。
- **AesRecon 共 9,071 对严格对齐 poor/good 图像，其中 903 对作测试集**；数据需签署协议邮件申请，并非直接可公开下载。

## 核心摘录 3：AesThinker 规划（§4.1–4.2）

- 训练监督：依据配对图、可选 ASR 教学字幕/音频转写提示，以 GPT-5.2 蒸馏**七维有序**编辑步骤，Gemini 3 校验并反馈重写；SFT 微调 Qwen3-VL-8B。
- 七维从全局到局部：**画幅比例 → 构图/取景 → 相机视角 → 主体位置 → 姿态与动作细节 → 焦点/景深 → 色彩/光线**。
- GRPO-A 由当前策略同 prompt 采样多条动作计划，组内相对奖励优化。奖励由**格式、与参考意图对齐、创造性/可执行性**构成，权重论文设为 `(0.1, 0.5, 0.4)`；若计划凭空增加人物/对象/背景会受惩罚。

## 核心摘录 4：AesEditor（§4.3）

- 基于 Qwen-Image-Edit-2511：以 (poor 图、good 图、编辑动作) 三元组训练动作条件编辑器；冻结多模态编码器与 VAE，仅对 MMDiT 做 LoRA。
- 使用 rectified-flow/flow-matching 训练目标将动作指导落实为像素级的结构变化；推理时先规划动作，再以原图+动作生成编辑结果。

## 核心摘录 5：实验、消融与证据边界（§5）

| 指标（AesRecon） | AesFormer | Qwen-Image-Edit-2511（无 Thinker） | Nano Banana Pro |
|---|---:|---:|---:|
| GPT-4o：相对 poor 的偏好胜率 | 65.33% | 16.50% | 54.44% |
| 人类：相对 poor 的偏好胜率 | 68.63% | 9.80% | 72.55% |
| Q-ALIGN 分数 | 3.51 | 3.20 | 3.24 |

- Nano Banana Pro 只在随机 **10% 测试子集**评测（API 成本），**不应将上表视作严格同样本对照**。
- 消融：仅 SFT 的相对 poor GPT-4o 胜率 61.04%，增加 GRPO-A 与编辑器后达到 65.33%；打乱七维顺序的 SFT 为 58.69%，说明顺序有用。
- 另有 15 名人类标注者在子集上进行偏好对照；自动审美评分与人工评价不可混为同一含义。

## 开源与复现边界

官方仓库已公开训练配置与测试入口：Stage 1(a) SFT（LLaMAFactory）、Stage 1(b) GRPO-A（EasyR1）、Stage 2 编辑器 LoRA（musubi-tuner），以及两阶段评测。README 说明权重已发布到 Hugging Face，但 AesRecon 原始图像仍需签协议获取；脚本有用户须改写的本地模型路径，复现结果需另行运行验证。

## 主要局限（§5–6、README）

- 只对**单张照片**的结构美学重建评测；多图与视频扩展属于未来工作。
- 生成式编辑的**身份/语义保持**是目标和约束，不能由美学胜率直接推出绝对无偏差。
- 训练基础模型及资源较重（论文实验使用 10 张 A40 48GB）；GPT-4o 评判有模型依赖，官方 README 亦提醒评测 API 映射不稳定会影响 win-rate 重现。

## 原始资料

- [论文全文（arXiv v1 HTML）](https://arxiv.org/html/2605.22126v1)
- [论文 PDF](https://arxiv.org/pdf/2605.22126)
- [作者代码与 README](https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026)
