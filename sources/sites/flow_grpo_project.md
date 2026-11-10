# Flow-GRPO 官方项目主页核验（2026-11-10）

- **项目：** Flow-GRPO: Training Flow Matching Models via Online RL
- **项目主页：** https://gongyeliu.github.io/Flow-GRPO/
- **论文：** https://arxiv.org/abs/2505.05470
- **代码：** https://github.com/yifan123/flow_grpo
- **预训练基座：** Stable Diffusion 3.5 Medium；仓库扩展支持 FLUX、Qwen-Image、Wan2.1 等
- **模型权重：** 主页 Model 指向 https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO ，官方 README 分列 GenEval / Text Rendering / PickScore 对应 Hugging Face 模型
- **可视化：** 主页展示三项任务的 Gallery、训练演化和生成结果包（GenEval / OCR / PickScore）
- **开放结论：** **已开源**。项目页 Code 跳转作者仓，README 提供环境与训练入口；官方 README 的后续 Flow-GRPO-Fast、GRPO-Guard、CPS 等为**后续更新**，不要混同 2025 原论文方法。
- **核对范围：** 逐项核对官网首页 Paper / Code / Model、三组任务 gallery / training process / result download 的入口；检查官方 README 的 quick start、模型与奖励服务器说明、近期变更。
- **通读：** 已完成

## 与代码、论文归档互链

- [论文 v5 阅读记录](../papers/flow_grpo_arxiv_2505_05470_reading.md)
- [官方仓代码分析](../repos/flow_grpo_official.md)
- [知识库项目实体](../../wiki/entities/paper-flow-grpo.md)

## 风险

项目页图像提升来自文字到图像任务；并不证明用于机器人流匹配动作策略时具有同等幅度收益。原始论文指标应以 arXiv v5 为准；仓库现在增加的工程变体应与论文主体区分。
