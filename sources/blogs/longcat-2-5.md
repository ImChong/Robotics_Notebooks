# 美团 LongCat-2.5 技术博客（2026-11-02）

- **标题：** LongCat-2.5 官方技术博客
- **来源：** [美团技术团队原文](https://tech.meituan.com/2026/11/02/Meituan-LongCat-2.5.html)
- **类型：** 官方技术博客
- **发布日期：** 2026-11-02
- **入库日期：** 2026-11-10
- **核对范围：** 官方技术博客的模型架构、视觉 token 化、Agentic RL、在线蒸馏、长上下文训练、量化与部署测量；与官方模型卡交叉核验。
- **通读：** 已完成

## 原文关键信息（提炼，非原文转载）

1. **1.6T 总参数 / 每 token 约 48B 激活参数。** LongCat-2.5 是原生多模态 MoE，切勿将总参数与单次 token 激活量混淆。
2. **视觉离散化：** Semantic-and-Aligned Encoder（SAE）配合 Residual Vector Quantization（RVQ），在原生分辨率上把视觉内容表示为离散 token；视觉 token 可离线缓存供训练复用。
3. **长程 Agent RL：** 针对真实软件工程任务训练长轨迹工具使用；强调部分 rollout 的异步过采样，以及处理策略版本不同步所引入的训练偏差。示例包括 GitHub PR/issue 与 F2P/P2P 测试反馈。
4. **多教师在线策略蒸馏：** 对多个教师的分叉续写进行并行采样和策略对齐，树状注意力减少重复前缀计算；这不是机器人运动控制中的多专家动作蒸馏。
5. **超长上下文训练：** 官方描述百万 token 上下文训练，配合混合并行与训练/推理 logprob 一致性优化；“百万 token”不代表每种部署都能直接承担相同的吞吐/时延。
6. **W4A4：** 官方报告相对 W8A8 的特定测试配置下 prefill 吞吐 +43.3%、decode 吞吐 +42.77%、TTFT -15.91%、TPOT -19.72%；属于厂商披露的测量，不能外推到其他硬件/负载。

## 评测与边界

- 模型卡汇报 Terminal Bench 2.1、Vision2Web、3DCodeBench、LongCatClawBench 等指标；对照需要核验数据集版本、评测预算与是否为自测。
- “Agent 可处理软件任务”不等于它已是可直接控制真实机器人的 VLA 策略。
- 官方披露的量化收益是特定部署环境对比，不是独立通用 benchmark。

## 对 wiki 的映射

- [LongCat-2.5 主实体](../../wiki/entities/longcat-2-5.md)
- [LongCat-Next 离散多模态研究](../../wiki/entities/paper-longcat-next.md)
