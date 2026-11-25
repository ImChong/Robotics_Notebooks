# Google Research：TimesFM-3 — A zero-shot foundation model for multivariate forecasting

> 官方一手资料归档

- **标题：** TimesFM-3: A zero-shot foundation model for multivariate forecasting
- **类型：** blog（Google Research 技术发布）
- **作者：** Ayush Jain、Rajat Sen；致谢 Yichen Zhou、Petros Mol、Abhimanyu Das、Samet Oymak
- **机构：** 谷歌研究院（Google Research）
- **发表：** 2026-08-31
- **原文：** <https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/>
- **代码：** <https://github.com/google-research/timesfm>
- **权重：** <https://huggingface.co/google/timesfm-3.0-pytorch>
- **入库日期：** 2026-11-21
- **核对范围：** 通读 Google Research 原始文章的动机、TimesFM 2.5 对照、模型规模与预训练数据量、patch/attention/CPM 方法、营销示例、GIFT-Eval/FEV-Bench/TIME 排名表述与结论；配合官方仓库核对可复现入口及授权边界。
- **通读：** 已完成

## 原文可核对事实

1. TimesFM-3 为 **330M** 参数，预训练覆盖 **超过 1 万亿时间点**（真实 + 合成），主张无需任务专用微调即可零样本预测。
2. 在 2.5 之前，TimesFM 主干主要针对单变量；3.0 可联合建模多目标、历史协变量及历史+未来已知协变量。
3. 每条序列分成 **32 时间步 patch**，单序列尺度归一化；交替使用 **因果时间注意力** 与 **跨变量全注意力**。
4. 使用 **Contiguous Patch Masking（CPM）**：未来预测区段的目标值掩码，已知未来的协变量可见，从而单次前向预测整段 horizon，而非逐 patch 自回归。
5. 各预测步输出 **0.1–0.9 共 9 个分位数**；0.1 / 0.9 可表述为相应分位数范围，不能未经验证宣称具有 95% 统计覆盖率。
6. Google Research 自测报告：在 GIFT-Eval、FEV-Bench、TIME 的预训练基础模型对照中，点预测与概率预测的平均排名居前/第一；**属于官方报告**，不是对所有可能数据集的普遍保证。
7. 官方以冰淇淋销量、关联商品与促销计划示例说明多变量条件预测；销售、客流等业务场景可映射到输入序列，但仓库未给出每个行业的单独精度保证。

## 版本与迁移边界

- TimesFM 2.5：**200M** 参数、最大 **16k 上下文**、可选 30M 分位数头；协变量通过 XReg 支持。
- TimesFM 3.0：**330M**、原生多变量/动态协变量、9 分位数、非自回归 horizon 预测；本资料不把 3.0 的特点回填到 2024 年原论文。
- 基础论文（ICML 2024）：<https://arxiv.org/abs/2310.10688>，属于家族先行工作，并非 TimesFM-3 新论文。

## 对 wiki 的映射

- [TimesFM 3.0 项目实体](../../wiki/entities/timesfm-3.md)
- [源码核查](../repos/google-research-timesfm.md)
- [Google Sheets 官方帮助](../sites/google-connected-sheets-timesfm.md)
