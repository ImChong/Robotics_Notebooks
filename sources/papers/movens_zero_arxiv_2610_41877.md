# Movens-Zero: Scaling Humanoid Motion Foundation Models with Large-Scale Human Videos

> 来源归档（paper；同一研究工作仅对应 `wiki/entities/paper-movens-zero.md`）

- **标题：** Movens-Zero: Scaling Humanoid Motion Foundation Models with Large-Scale Human Videos
- **作者：** Wendong Bu、Tinghong Ye、Qizhou Wang 等（浙江大学、宇树科技等）
- **版本：** arXiv:2610.41877v1（2026-10-30，技术报告 / 预印本）
- **HTML 全文：** https://arxiv.org/html/2610.41877
- **PDF：** https://arxiv.org/pdf/2610.41877
- **项目页：** https://movens-zero.github.io/
- **代码：** https://github.com/wendell0218/Movens-Zero
- **数据：** [Movens-5M 官方数据卡](../datasets/movens-5m.md)
- **基准：** https://huggingface.co/datasets/wendell0218/Movens-Intent
- **入库日期：** 2026-11-03；2026-11-23 基于一手资料复核升级
- **核对范围：** 通读 arXiv HTML v1 全文的摘要、第 1–7 节、表 3–9、附录 A–G（数据管线、重定向、模型、训练、指标、失败案例）；对照项目主页、HF 数据卡与官方 GitHub 训练及推理入口。
- **通读：** 已完成

## 研究问题

大规模互联网人体视频有动作种类和场景覆盖，却缺机器人可跟踪的状态/动作；动捕规模有限，人—机形态差异和物理失真会令直接学习失败。本文用 Movens-5M 数据引擎、Movens-Intent 测评和 Movens-Zero 多模态动作生成模型形成数据→学习→G1 动作跟踪链路。

## 数据引擎（论文第 3 节 / 附录 A–B）

1. **Clip generation**：从互联网视频剪切连贯运动片段。
2. **Quality filtering**：质量/人体动作筛选。
3. **Semantic annotation**：动作类别、场景、人体与物体交互说明；对齐文本、视频、语音、音乐。
4. **Motion recovery**：从视频恢复人体运动（文中包括 GVHMR、PromptHMR 和 SMPL-X 相关处理）；检查接地、滑移、穿透、抖动等并修复。
5. **Physical refinement**：对目标人形进行运动学初始化、物理约束的轨迹优化和多套闭环跟踪策略执行验证，剔除不可跟踪样本；报告 6 类物理检查与 4 个 tracking policies。

官方数据规模：15M human motions、5M robot-executable motions、>10k h、1.08B+ frames、297 motion types、16 scene groups。**统计口径不等同于中央 Hugging Face 预览文件的下载量或逐条真机验收数**。

## 模型与训练（论文第 5 节 / 附录 C–D）

- 多模态 Qwen2.5-Omni 理解输入 + 条件 flow matching 的动作专家，统一文本、RGB 视频、语音、音乐。
- 采用机器人中心表征和包含状态历史的分块动作预测；七任务混合训练，支持不同模态条件和数据权重。
- MM-RTC（multimodal real-time chunking）通过前一次动作 chunk 的重叠前缀与更新的条件、状态，降低分块切换不连续。
- 目标动作/评测主本体为 Unitree G1；官方代码支持 36 维绝对姿态（29 joints + 根平移 3 + 四元数 4）及 40 维 root-centric 分块表达。两者不可混作相同 schema。

## 实验（论文 `4/`6、表 7–9）

- Movens-Intent：8,192 个**source-disjoint、专家检查**的条件–动作配对，文本、语音、音乐和视频各 2,048。
- 固定 XL 动作专家，Movens-5M 从 1% 扩到 100%：Text R@3 **12.72→62.57%**、Speech R@3 **13.98→74.30%**、Video MPJPE **132.82→40.90 mm**、Tracking Score **0.21→0.72**。这是**论文的受控对比**，不代表零样本真实工厂成功率。
- 在满量数据上 S（64.2M）→XL（910.5M）专家，Text R@3 **10.30→62.57%**；分辨数据规模与模型容量收益。
- 表 9 分阶段数据整理消融：从原始恢复动作到完整过滤/修复/跟踪/分布平衡，Text R@3 **48.72→62.57%**、Tracking Score **0.41→0.72**；说明不仅要看语义相似，还要核查动作可跟踪性。

## 限制及开源口径

- 论文附录 F.3 列出高动态动作和大幅移动相机条件下失败案例。
- 官方公开 [推理/训练/评测代码及权重入口](../repos/movens-zero.md)；推理、训练命令应区分。「有代码」并非「已在本环境跑过 GPU/真机」。
- 官方数据卡将中央仓库定义为 Parquet 预览，完整数据链接指向四个 tar 分片仓库；许可为 `other`，不可预设可自由商用，详见 [数据来源归档](../datasets/movens-5m.md)。

## 对 wiki 的映射

- [唯一论文实体：Movens-Zero / Movens-5M](../../wiki/entities/paper-movens-zero.md)
- [具身数据路线](../../roadmap/depth-embodied-data.md)
