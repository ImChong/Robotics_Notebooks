# Awaken-HI 官方项目主页（EngineAI）

> 一手 site 归档，2026-11-21 核查。

- **类型：** site
- **项目页：** <https://engineai-robotics.github.io/Awaken-HI/>
- **论文：** <https://arxiv.org/abs/2611.26420>
- **官方仓库 / Code 按钮：** <https://github.com/engineai-robotics/Awaken-HI>
- **数据 / Data 按钮：** <https://huggingface.co/datasets/EngineAI-Embodied/Awaken-Human>
- **入库日期：** 2026-11-21
- **对应详情：** [Awaken-HI](../../wiki/entities/paper-awaken-hi.md)
- **对应论文精读：** [arXiv v1](../papers/awaken_hi_arxiv_2611_26420.md)
- **开源状态：** 论文公开；官方仓库只有说明/媒体，目前无训练或推理实现、无权重；HF 仅开放样例录制，非完整 50,000 h。

## 核对结果

1. 官方页展示预训练数据约 50k h、100+ 技能、17.02M episodes、5.41B 姿态帧、11.18M 指令。
2. 方法部分为 Qwen3-VL + 共享 DiT action expert，训练 40D T800 action、161D 原始人类动作、18D camera wrist，附加场景深度与 subtask 监督。
3. 5k→50k h 时动作 MSE 下降约 11%，三个家务任务从 56.7%→90.0%；主评测八任务平均成功率 93.75%，工业 / 家庭长程 90% / 80%。
4. 页面 Code 按钮虽然可访问 GitHub，但**不能等同于可运行代码已开源**。以官方 README TODO 与实际文件目录为准。
