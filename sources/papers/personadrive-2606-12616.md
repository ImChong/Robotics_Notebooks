# PersonaDrive: Human-Style Retrieval-Augmented VLA Agents for Closed-Loop Driving Simulation
> arXiv 论文来源归档（版本 v2）
- **论文身份：** arXiv:2606.12616
- **中文标题：** PersonaDrive：人类驾驶风格检索增强 VLA
- **原文：** https://arxiv.org/abs/2606.12616v2
- **HTML：** https://arxiv.org/html/2606.12616v2
- **PDF：** https://arxiv.org/pdf/2606.12616v2
- **策展：** [多模空间2026.11上半月](../blogs/duomo_embodied_2026_11_first_half.md)
- **代码：** https://github.com/PervasiveAutonomyLab/PersonaDrive
- **核对范围：** 指定版本的标题、摘要及核心方法段落；完整实验表、必要附录、发布状态尚未逐节核对。
- **通读：** 未完成
## 核心摘录（方法级，不等于全文核验）
- **研究问题：** 同一导航路线可能有保守、适中和激进等行为偏好，风格不应等同于完全独立的策略模型。
- **主要机制：** 15 名驾驶者多风格 CARLA 示范 → 训练场景状态检索头与分风格索引 → 根据目标风格检索相似片段 → 示范进入 VLA 上下文 → 生成航点并在 CARLA 闭环执行。
- **实验范围：** 官方仓库称 15 名驾驶者、29,777 帧，Bench2Drive 无风格 Driving Score 88.95；均属作者报告，未执行 CARLA 复现。
- **主要限制：** 验证主要在 CARLA/Bench2Drive 仿真，索引覆盖、检索错误及安全约束不能直接推及真实道路。
## 后续待核
完整通读论文实验、附录与局限；如有代码则完成复现环境与关键入口检查。官方成果与独立复现需要区分。
## 对 wiki 的映射
- [PersonaDrive：人类驾驶风格检索增强 VLA](../../wiki/entities/paper-personadrive-2606-12616.md)
