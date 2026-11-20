# PersonaDrive: Human-Style Retrieval-Augmented VLA Agents for Closed-Loop Driving Simulation
> arXiv 论文来源归档（版本 v2）
- **论文身份：** arXiv:2606.12616
- **中文标题：** PersonaDrive：人类驾驶风格检索增强 VLA
- **原文：** https://arxiv.org/abs/2606.12616v2
- **HTML：** https://arxiv.org/html/2606.12616v2
- **PDF：** https://arxiv.org/pdf/2606.12616v2
- **策展：** [多模空间2026.11上半月](../blogs/duomo_embodied_2026_11_first_half.md)
- **代码：** https://github.com/PervasiveAutonomyLab/PersonaDrive
- **核对范围：** 已通读 arXiv:2606.12616v2 正文 §1–§5、主要表格和附录 A–L，包括三阶段训练、数据分割、检索失败模式、表 S4 超参、延迟/深度消融、prompt 和术语表；另静态检查官方仓库。
- **通读：** 已完成
## 原文核验要点
- **数据**：15 人、三风格、55 路线/29,777 帧，测试 220 条 Bench2Drive 路线相互分离。
- **流程**：图像和文本 triplet mining → 视觉/控制检索头 → 按风格 FAISS → K=2 in-context 示例 → SimLingo VLA 航点预测。
- **报告**：无风格 DS 88.95 vs SimLingo 85.07，SR 72.29 vs 67.27；K=2 推理约 557.7ms/步 vs K=0 的 480.3ms，舒适性 28.09 vs 33.67。
- **范围**：仅 CARLA 闭环模拟，未验证真实道路。

## 后续待核
原文已核对完毕。作者实验和方法均为**论文报告**，未独立运行仿真/真机或大型 GPU 训练；代码资源以源码归档记录的实际可访问状态为准。

## 对 wiki 的映射
- [PersonaDrive：人类驾驶风格检索增强 VLA](../../wiki/entities/paper-personadrive-2606-12616.md)
