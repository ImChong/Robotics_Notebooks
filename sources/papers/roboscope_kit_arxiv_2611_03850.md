# RoboScope Kit：统一接口与快速可扩展具身评测

- **标题：** RoboScope Kit: Unified Interfaces for Fast and Scalable Embodied-AI Evaluation
- **类型：** paper
- **版本：** arXiv:2611.03850v1，2026-11-02
- **原文：** https://arxiv.org/html/2611.03850v1
- **核对范围：** 正文 1–8、附录 A–E（接口、实现、评测、基线版本、完整计时、故障边界）；不是摘要或公众号替代全文。
- **通读：** 已完成
- **论文：** https://arxiv.org/abs/2611.03850
- **项目页：** https://mint-sjtu.github.io/RoboScope-Kit/
- **代码：** https://github.com/MINT-SJTU/RoboScope-Kit
- **核对日期：** 2026-11-07

## 核心摘录

- Policy/adapter 分离模型与基准；独立 K/g 池与实测并发搜索，保持原任务、初态、动作和评分契约。
- RoboTwin Easy 500 回合：21,582 s 单卡基线至 862 s 四卡优化，25.04 倍；同四卡 K4/g4 基线 5,934 s，软件与调度合计 6.88 倍。表 16–17 说明最快配置仅一次运行。
- 优化运行 K16/g4、stride 16、两回合工作项、clutter 元数据缓存，80 核与四张 48 GiB RTX 4090。成败和终止步全匹配三次单卡运行；231/500 成功，端到端 928 s。
- LIBERO-Long 使用另一 StarVLA checkpoint；MPS 搜索后 6.16 倍为评测阶段，含搜索端到端约 3.02 倍。成功率等效不代表逐回合相同。
- 附录 B.2/B.3 的 StarVLA 客户端版本、图像插值、动作反归一化和 RLinf 采样差异会影响比较；自动搜索有额外成本。

## 开源状态与映射

截至核对日，官方源码已开源（Apache-2.0），替代此前二手索引「未列代码」记录；仿真资产和模型检查点依赖上游。

- [统一实体节点](../../wiki/entities/paper-roboscope-kit.md)
- [项目页归档](../sites/roboscope-kit.md)
- [源码归档](../repos/roboscope-kit.md)
