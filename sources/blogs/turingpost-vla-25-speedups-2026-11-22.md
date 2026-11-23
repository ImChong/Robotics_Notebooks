# 25 Ways to Make VLA Robots Faster（Turing Post 2026-11-22 原文阅读归档）

- **类型：** blog / 跨论文技术综述（Turing Post 原始策展文章，并非 25 篇原论文）
- **作者：** Alyona Vert. / Turing Post
- **发表日期：** 2026-11-22
- **原文链接：** https://www.turingpost.com/p/ways-to-make-vla-robots-faster
- **核对范围：** 2026-11-23 通读原文的导语、25 项方法概览、逐项说明、指标解释、VLALagBench / VLA-Triage 评测建议及 FAQ；逐条打开其 arXiv 引用入口，确认论文编号与标题。
- **通读：** 已完成
- **阅读边界：** 「通读」只指 **Turing Post 完整博客原文**，不代表已阅读全文、运行代码或复现全部 25 篇原始论文。
- **核心观点：** 先定位耗时环节：单模型调用、块间等待、动作解码或网络/调度，才有意义谈加速。
- **入库日期：** 2026-11-23

## 25 种方法与论文定位（归纳，不转载原文）

| # | 方法 / 原论文 | 瓶颈层 | 方法概述 | 来源报告结果 | 计量/限定 |
|---|---|---|---|---|---|
| 01 | [APS-VLA](https://arxiv.org/abs/2611.13030) | 模型内部 | 选择动作相关视觉 token | 137.25→93.25 ms；真机平均成功率 60%→61% | 真机策略推理 |
| 02 | [ActPrune](https://arxiv.org/abs/2611.08827) | 模型内部 | 动作路径引导的视觉 patch 剪枝 | OpenVLA-OFT 1.61×；π0.5 仅 1.03× | LIBERO；不同骨干 |
| 03 | [EfficientVLA](https://arxiv.org/abs/2506.10100) | 模型内部 | 层、视觉 token、去噪特征重用 | 1.93×；成功率 −0.6 pp | SIMPLER / A40 |
| 04 | [VLA-Cache](https://arxiv.org/abs/2502.02175) | 模型内部 | 跨帧 KV 缓存 | CUDA 推理最高 1.7×；控制频率 +15% | 两项指标不能互换 |
| 05 | [QuantVLA](https://arxiv.org/abs/2602.20309) | 模型内部 | W4A8 后训练量化 | 两个模块显存 4.27→1.28 GB | 显存而非整体速度 |
| 06 | [SmolVLA](https://arxiv.org/abs/2506.01844) | 模型内部 | 轻量化模型与少视觉 token | 约 450M 参数、每张图 64 视觉 token | 架构约束，非单独加速倍数 |
| 07 | [RTC](https://arxiv.org/abs/2506.07339) | 异步执行 | 推理/运动重叠，冻结已提交动作前缀 | 真机模型调用 76→97 ms；可改善任务吞吐 | 单次推理反而变慢 |
| 08 | [TT-RTC](https://arxiv.org/abs/2512.05964) | 异步执行 | 训练时加入提交前缀及延迟 | 远程 H100 请求端到端 135→108 ms | 与 RTC 对比 |
| 09 | [SR-VLA](https://arxiv.org/abs/2611.22356) | 异步执行 | 预测未来观测、选择性缓存修复 | 剩余等待 128→48 ms | LIBERO 仿真；非整体推理延迟 |
| 10 | [DASH](https://arxiv.org/abs/2611.11854) | 异步执行 | 慢 VLM 与快动作头分时钟 | 0.8B–8B 维持 20 Hz；同步 9.7→4.1 Hz | 延迟感知 RoboTwin |
| 11 | [PTS](https://arxiv.org/abs/2611.01390) | 异步执行 | 路径不变，调整沿路径的执行速度 | 任务吞吐 π0.5 +28%、SmolVLA +38% | LIBERO 异步对照 |
| 12 | [RAVEL](https://arxiv.org/abs/2609.34170) | 异步执行 | 滚动去噪缓冲 | 动作可用时延 272.86→93.45 ms | 4090 同步仿真 |
| 13 | [VLASH](https://arxiv.org/abs/2512.01031) | 异步执行 | 根据动作抵达时刻预测未来状态 | 理论最差反应延迟 546.4→46.4 ms | 基于测量推算，非实际响应分布 |
| 14 | [A2C2](https://arxiv.org/abs/2509.23224) | 异步执行 | 慢策略计划叠加实时纠正头 | 仿真成功率 64.4%→84.2% | LIBERO Spatial 指定延迟设置 |
| 15 | [Zora](https://arxiv.org/abs/2611.11098) | 动作生成 | 历史动作检索与批量验证 | 软件最高 3.4×；硬件协同 7.6× | Thor vs 模拟硬件不能混用 |
| 16 | [VSpark](https://arxiv.org/abs/2611.22666) | 动作生成 | 自回归动作 token 推测解码 | 解码 1.28–2.10× | 四种 AR VLA；LIBERO |
| 17 | [UAD](https://arxiv.org/abs/2609.37772) | 动作生成 | 当前动作优先、后续逐步去噪 | 动作可用 21.6→11.4 ms | SmolVLA / LIBERO-Plus |
| 18 | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | 动作生成 | 连续动作块并行解码 | 动作生成吞吐 26×；单次 239.6→72.9 ms | 单图 LIBERO；非完成任务 26× |
| 19 | [SnapFlow](https://arxiv.org/abs/2604.05656) | 动作生成 | 渐进自蒸馏成一步 Flow | Batch1 延迟约 /3.3；98.75% 成功率 | LIBERO；与直接一步基线比较 |
| 20 | [FlashVLA](https://arxiv.org/abs/2608.27384) | 动作生成 | 连续动作块交错噪声流式解码 | 真机单 GPU ≥30 Hz | 稳态控制频率，不含预热 |
| 21 | [VLA-ULAP](https://arxiv.org/abs/2609.18663) | 云边协同 | 远程调用间用本地动作预测 | 每成功任务远端请求 −70%～72% | SO-101 真机；非耗时 |
| 22 | [Robion](https://arxiv.org/abs/2609.12075) | 云边协同 | 多机器人多阶段请求调度 | 4 GPU / 64 模拟机器人；98% 截止时间满足 | 模拟负载、不是 64 台真机 |
| 23 | [VLA Handover](https://arxiv.org/abs/2611.08987) | 云边协同 | 本地小模型优先、失败时转云端 | 三套 LIBERO 请求 −49%～77%；Long 成功率 63.6% vs 93.2% | 任务分布和成功率权衡 |
| 24 | [CloudEdgeVLA](https://arxiv.org/abs/2608.00569) | 云边协同 | 云端语义，本地新鲜视觉与动作头 | 1 s 人为 RTT 下移动目标真机 7/10 vs 0/10 | Franka 小样本测试 |
| 25 | [ComVLA](https://arxiv.org/abs/2609.07838) | 云边协同 | 本地编码，按带宽预算传视觉 token | 32 token LIBERO 95.4%；基线 512 token 96.9% | 计算延迟不含网络 |

## 评测维度与易误读点

- **模型调用时延（model inference latency）**：RTC 真机从 76 ms 增至 97 ms，不能以这个数判断其整个系统更慢；其作用是与已有动作并行重叠。
- **动作可用时延（action availability latency）**：RAVEL / UAD 减少首个动作出炉时间，不直接推出物理反应时间。
- **控制频率（control frequency）**：DASH 的 20 Hz、FlashVLA 的 ≥30 Hz，测量平台、异步模式与预热条件不同。
- **动作生成吞吐（action-generation throughput）**：OFT 的 26× 是每秒产生动作的数量，不能按任务完成时间解释。
- **网络调用量、服务容量、显存**：ULAP、Robion、QuantVLA 分别使用不同的因变量；显存降低并不自动提升真机节拍。
- **部署测量建议**：同任务/机器人/硬件记录观测时间戳到真正消费它的动作、P50/P95 延迟、任务完成时间、成功率；远端部署再记网络 RTT/排队长尾。
- **辅助评测工作：** [VLALagBench](https://arxiv.org/abs/2611.03103)（任务延迟敏感性）、[VLA-Triage](https://arxiv.org/abs/2611.06710)（不同硬件与优化组合的瓶颈）。
- **来源层级：** 表中结果是 Turing Post 对各论文的策展转述；已逐项验证 arXiv 编号可对应，但**没有逐篇全文重算所有表格**。需要论文级证据时进入对应的独立论文详情节点或论文原文。

## 对 wiki 的映射

- [VLA 加速 25 法工程对照](../../wiki/comparisons/vla-acceleration-25-methods-2026.md)
- [控制与推理频率解耦](../../wiki/concepts/control-inference-frequency-decoupling.md)
- [VLA 纵深学习路线](../../roadmap/depth-vla.md)
