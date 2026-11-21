# Thinking Machines Lab：Introducing Inkling 2 Mini

- **官方原文：** https://thinkingmachines.ai/news/inkling-2-mini/
- **标题：** Introducing Inkling 2 Mini
- **机构：** Thinking Machines Lab
- **发布日期：** 2026-11-20
- **入库日期：** 2026-11-21
- **类型：** 官方技术博客（非同行评审论文）
- **模型卡及权重：** https://huggingface.co/thinkingmachines/Inkling-2-Mini
- **核对范围：** 博客全篇：Capabilities/Limitations、Architecture and training（Looping design、NVFP4、Pruning and healing、Training）、Monitorability（No-CoT、CoT monitorability、Agentic monitorability、Mechanistic interpretability）、Try Inkling 2 Mini；逐节核对正文数值与测试设置。
- **通读：** 已完成

## 技术摘录（概括而非原文复制）

1. **共享权重循环：** 24 层分 4 个 6 层 block；头尾各一次，中间 12 层循环 3 次。attention residuals 使每次 loop 使用 prelude 输出与上一遍 message 的可学习门控混合；共 24 层参数、48 次层应用/token。
2. **MoE 剪枝与修复：** 原始每层 384 专家、每 token 选 8；midtraining 一次删去三分之二，余 128；再以 1T tokens 蒸馏恢复。其后 SFT 和约 2.2T tokens / 2,000 RL steps 的后训练；这是公开方法轮廓，**不是**开放训练数据或可执行训练脚本。
3. **原生低精度：** 专家层 NVFP4 贯穿预训练到 RL，20 层 NVFP4 + 末 4 层 MXFP8；约 20GB 权重存储。循环增加计算深度，不代表增大静态存储参数。
4. **性能声明：** DeepSWE v1.1 48.2%、Vals Index v2.2 46.5%、BrowseComp 79.0%（compaction）；所有数字为发布方报告，跨模型 harness 口径不全相同。
5. **循环收益不等于知识增长：** 变量循环研究模型 TML-d24-varloop-research 的 no-CoT 50% time horizon 从 1 pass 的 16 s 升至 3 pass 的 37 s，超过 3 pass 常递减；多跳查找仍可受益。此实验**不是固定三遍 Inkling 2 Mini 可动态切换循环**。
6. **可监控性研究：** 变量循环模型的正确 GPQA Diamond 题目上，1 pass / 5 pass 平均 CoT 约 3703 / 1661 tokens，简单可读性代理指标无明显恶化；LinuxArena 隐蔽副任务评测未发现随循环遍数单调变差的规律。研究范围、RL 仅按 3 passes 训练等限制需保留。
7. **限制：** MultiChallenge 50.0%、ComplexConstraints v1.1 33.4%；长多轮、事实回忆、中文质量、危险工具操作需要额外验证，不适合直接接入安全关键决策链。

## 工程含义与证据边界

- 模型适合研究“每 token 更多内部串行计算”与 token 长度、吞吐、能耗之间的取舍，但不是机器人动作 head、低层控制策略或 WAM。
- 用于机载语义规划时，仍须测图像输入时延、最坏推理周期、显存和系统工具安全；发布方未提供机器人控制回路实测。
- 官方博客给出了推理环境/配套模型入口，未给出可独立复现全量预训练的端到端训练工程。

## 对 wiki 的映射

- [Inkling 2 Mini 模型实体](../../wiki/entities/inkling-2-mini.md)
- [模型架构纵深路线](../../roadmap/depth-model-architecture.md)
