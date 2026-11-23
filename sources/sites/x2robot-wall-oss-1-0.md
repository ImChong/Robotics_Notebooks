# WALL-OSS-1.0：Understand More, Act Better（知深，行远）

> 自变量机器人（X Square Robot）官方技术博客原文归档；本文不是 arXiv 论文，不能用前代 WALL-OSS / WALL-OSS-0.5 的论文编号替代。

- **标题：** WALL-OSS-1.0: Understand More, Act Better / WALL-OSS-1.0：知深，行远
- **类型：** 官方技术博客 / 技术报告说明
- **机构：** X Square Robot Team（自变量机器人）
- **发布日期：** 2026-11-23
- **版本：** 官方博客发布版（2026-11-23）
- **官方正文：** <https://x2robot.com/blog/wall-oss-1>（英文 <https://x2robot.com/en/blog/wall-oss-1>）
- **配套 API 文档：** [Wall-X API 接入说明](x2robot-wallx-api.md)；该文档并未明确给出 WALL-OSS-1.0 后端版本
- **既有代码参考：** <https://github.com/X-Square-Robot/wall-x>（当前 README 明确以 WALL-OSS-0.5 和更早版本为主，不能据此宣称 1.0 已发布实现）
- **核对日期：** 2026-11-23
- **核对范围：** 官方中英文博客的原生 VLA 预训练、动作专家与 RTC、Lite/Nano/Gen/AR 架构、开放式真机 / ManipArena / X2Real 测试、XR Harness 长程执行与引用信息；并核对 wall-x 官方 README 是否列出 1.0 可运行代码
- **通读：** 已完成

## 原文核心机制（归纳，不转存全文）

1. **Stage 1 原生 VLA：** 从 Qwen3.5 基座起步，把语言、图像/视频和机器人轨迹统一为序列预测；改进版 X-Tokenizer 将多本体连续轨迹编码成离散动作词元，动作 tokenizer 预训练后冻结，视觉编码器、投影和主干联合训练。
2. **Stage 2 动作专家：** 延伸为适合精准连续控制的动作生成方案；利用多形式监督（状态预测、状态和控制预测、正逆动力学）与共享上下文分块，使用 flow 生成动作块；RTC 利用无噪声未来命令前缀稳定相邻 chunk 的拼接。
3. **模型家族：** Lite（4B）降低适配成本，Nano（2B）降低计算成本；Gen 引入 Generation Expert 和仅训练期使用的 WAM，从第一人称视频等低质量/缺失动作标注来源获得视觉预测监督；AR 使用分层残差动作编码、时间优先自回归解码与增量执行，无需额外 flow/diffusion 动作专家。
4. **推理系统：** Nano 配合 XInfer 在生产场景评估中，一块 Blackwell 消费级 GPU 服务 18 台机器人；文中还讨论 Jetson Thor 等机载优化，但不可将“18 台”泛化到所有 GPU/工作负载。
5. **长程执行：** XR Harness 将 System 2（规划与重规划）、System 1（WALL-OSS-1.0 动作策略与完成检查）和 System 0（RAP 与硬件）串起来；这是上层编排带来的长程能力，不等于 VLA 模型单独完成全部规划。

## 评测原始口径

| 官方实验 | 主要结果 | 不能混同的条件 |
|---|---|---|
| 开放式真实机器人评测 | **312** 个评估者自由选择的任务，零样本成功率 **55.1%**，平均任务评分 **6.45/10** | 同一预训练策略、不做任务专属微调；不属于固定 ManipArena 主榜 |
| VLA–VLM 通用能力 | Stage 1 **79.2%** vs 原 Qwen3.5 基座 **79.4%** | 8 个 benchmark 的主分数均值；不同模型协议可能不同 |
| X2Real 桌面仿真 | 总分 **67.55/100** vs Lingbot-VLA-2.0 **40.94** | 除官方注明的特殊基线外，参评模型均对仿真数据微调一个 epoch；并非零样本真机结果 |
| X2Real 移动操作 OOD | **44.17** vs π0.5 **24.83** | 三项移动操作任务平均分 |
| Nano 服务吞吐 | Blackwell 消费级 GPU **18 台机器人** | 生产部署场景 + XInfer 优化，不能推定单机控制环时延 |

## 公开资源与复现边界（截至 2026-11-23）

- 官方技术博客公开方法与基准描述；本文未见可直接下载的 WALL-OSS-1.0 专属代码、权重与预训练数据下载链接。
- 核对官方 [wall-x README](https://github.com/X-Square-Robot/wall-x)：News/Models 仍列至 WALL-OSS-0.5（代码与四个旧权重已公开）。**1.0 发布状态须与 0.5 区分**，不能把旧仓库改写为 1.0 的可运行实现。
- 用户提供的 [Wall-X API 文档](x2robot-wallx-api.md) 有可申请的云推理接口，其握手元数据示例为 `model: wall-x`，不披露后端模型版本，因此不能证明可调用 1.0。
- 目前未获取完整 1.0 训练实现/数据，也未独立复现实验；本文只记录官方披露与工程可验证接口。

## 对 wiki 的映射

- [WALL-OSS-1.0 独立实体节点](../../wiki/entities/wall-oss-1-0.md)
- [WALL-OSS-0.5 的独立报告节点](../../wiki/entities/paper-wall-oss-0-5.md)
- [WALL-OSS 初代与 wall-x](../../wiki/entities/cn-os-wall-x.md)
