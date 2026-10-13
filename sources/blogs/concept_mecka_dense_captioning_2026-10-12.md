# Frontier models can’t yet label robot training data accurately

> 来源归档

- **标题：** Frontier models can’t yet label robot training data accurately
- **类型：** blog / benchmark report（Concept，与 Mecka 合作）
- **链接：** <https://concept.dev/blog/dense-captioning>
- **发布日期：** 2026-10-12
- **入库日期：** 2026-10-13
- **一句话说明：** 在 200 段第一视角操作视频上对比通用前沿模型与 Mecka 专用标注模型，显示细粒度动作识别仍是主要错误来源，提示机器人数据标注需专用模型与人工复核。
- **开源状态：** 文章未链接评测代码、完整数据集或可运行实现；文中只说明部分实验使用 ffmpeg、Python、SAM 3.1 与 Grounding DINO。
- **沉淀到 wiki：** 是 → [`wiki/concepts/dense-captioning-for-robotics.md`](../../wiki/concepts/dense-captioning-for-robotics.md)

---

## 评测设置

- **数据：** Mecka 提供 200 段视频（来自 100 个源视频，每个取 2 段），中位时长 3.8 秒，总计约 13.5 分钟。参考标签按顺序标出每个动作、作用对象和执行手。
- **错误口径：** major error 包括动作错误、漏动作、多报动作、对象错误或左右手错误；minor error 包括复数、对象命名和动作精度问题。全正确要求 major/minor 均为零；“正确或仅有轻微问题”仅要求没有 major error。
- **通用模型：** GPT-6 Astra、GPT-6.1 Sol、Claude Opus 5.5、Claude Fable 5.1、Gemini 3.8 Flash。作者以不同推理档位评测，使用离线工具环境；GPT/Claude 经各自代码代理运行，Gemini 走 API。

## 主要发现

- GPT-6 Astra 在所测通用模型中最好，Max 档全正确率为 **43.5%**；动作错误是各模型最常见的 major error，单类错误在 **28.0%–39.5%** 的片段中出现。
- 将采样从 4 fps 提高到 16 fps，在全量 200 段上并未提高 Astra 的全正确率，尽管个别片段的动作步骤因此可辨。
- 仅做对象名纠错、或给画面标注手与物体，准确率各提高约 **0.5 个百分点**；先提供全视频摘要再逐段标注反而使准确率下降 **5 个百分点**。组合摘要与对象纠错时，纠错收益升至 **1.5 个百分点**。
- SAM 3.1 的手—物接触候选中，89.4% 的片段能找到至少一个真值对象，但仅 27.3% 排名第一；它可提供候选对象名，不能解决动作动词与步骤时序。
- Mecka 的模型标注在该集上全正确率为 **79.5%**，高于最佳通用模型的 43.5%；Mecka 表示仍以人机协作达到其 95% 目标精度。

## 解释边界

这是小规模、单一合作方数据上的报告，不等同于公开基准或可独立复现的论文。结果支持“当前通用模型不宜无审核地产生训练标签”，但不证明所有专用标注模型或所有机器人任务都能达到同等水平；文中 Mecka 标注评审也未设置与通用模型相同的 minor-error 分类。

## 对 wiki 的映射

- [`wiki/concepts/dense-captioning-for-robotics.md`](../../wiki/concepts/dense-captioning-for-robotics.md)：密集动作字幕的任务定义、评测边界与工程建议。
- [`wiki/concepts/robot-data-supervision-signal-types.md`](../../wiki/concepts/robot-data-supervision-signal-types.md)：补充第一视角视频语义标签与机器人动作监督不可混同。
