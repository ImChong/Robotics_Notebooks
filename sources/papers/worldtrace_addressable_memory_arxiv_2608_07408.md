# WorldTrace: Addressable Memory for Video World Models / WorldTrace：视频世界模型的可寻址记忆

- **类型：** 论文原文归档（NeurIPS 2026）
- **原始标题：** Addressable Memory for Video World Models
- **作者：** Xindi Wu, Sven Elflein, James Lucas, Olga Russakovsky, Laura Leal-Taixé, Despoina Paschalidou, Jonathan Lorraine, Aljosa Osep
- **机构：** NVIDIA（英伟达）、普林斯顿大学（Princeton University）、多伦多大学（University of Toronto）、Vector Institute（向量研究所）
- **论文：** https://arxiv.org/abs/2608.07408
- **HTML（v2）：** https://arxiv.org/html/2608.07408v2
- **PDF（v2）：** https://arxiv.org/pdf/2608.07408
- **项目页：** https://research.nvidia.com/labs/sil/projects/WorldTrace/
- **版本：** arXiv:2608.07408v2，2026-11-11；论文 2026-08-07 首次提交
- **入库日期：** 2026-11-23
- **核对范围：** arXiv v2 HTML 正文 §1–5（问题诊断、虚拟位置、规范化 key 聚合、landmark、实验及消融）、附录 A–I 中的注意力投影推导、长程/跨骨干评测、LoopBench 协议、Algorithm 1、硬件耗时及限制；对照 NVIDIA 官方项目页的更新版结果表与源码入口。
- **通读：** 已完成

## 问题设定

- 自回归视频世界模型依赖时间 RoPE 的 KV cache；生成超出训练时相对时间间隔后，即使旧 key 尚在，查询也可能难以定位该记忆（**addressability**）。
- 将不同时间 RoPE 相位的 key 在旋转后直接平均会抵消信息（**informativeness**）。可寻址性与信息保真度要同时处理。
- 本工作只改推理期缓存管理；**不重训生成器**，也不能据此推断机器人策略成功率提高。

## 论文机制（§2–3、Appendix H）

1. **Slot-rank virtual position**：本地窗口长度 `L_attn = N_s + N_r`，摘要槽 `s` 使用 `t_v(s) = q - (L_attn - 1 - s)`；当 `L_attn - 1 <= Δt_train` 时，所有摘要槽与当前 query 的相对位置保持在训练范围。
2. **Field**：先将原始 key 去除原时间戳的 RoPE 旋转，在 canonical 坐标内按组/计数平均，再按当前摘要槽虚拟时间旋回；value 本身不带 RoPE，可直接平均。
3. **Landmark**：在场景进入/切换时识别值得保留的帧，冻结该帧的 canonical key，读取时单次重旋到槽位虚拟位置；避免 bfloat16 多次去旋/重旋引起累积误差。
4. **Combined**：固定大小的近端真实帧窗口 + 远端槽，基于采样 query 的 attention-response error 与独特信息损失，选择合并、保留 exemplar 或淘汰；参见论文 Algorithm 1。这里的“landmark”是关键视觉记忆而非 SLAM 几何地标。
5. **预算**：MG2 主实验使用 `N_s=4, N_r=2`，局部关注 6 帧；训练相对时间范围最大 5 个 latent-frame steps。

## 实验数据与口径（§4、Appendix F–H）

- **模型/协议**：Matrix-Game-2 1.3B（带 3D-RoPE），LoopBench 4 类任务 × 各 3 个配置，共 12 条；17 个场景 × 3 seeds = 51 rollouts。比较第一次与返回位置生成帧，无外部参考视频。
- **指标**：DINOv2 scene consistency（越高越好，场景特征相似）和 Return LPIPS（越低越好，感知距离）；不是精确图像重建率，更不是物理仿真精度。
- **12 配置平均 DINOv2**：Sliding Window 0.587；WorldTrace-Field 0.576；WorldTrace-Landmark 0.715；组合 WorldTrace 0.778。组合相比滑窗的相对提升 `(0.778 / 0.587 - 1) ≈ 32.6%`，不是 +32.6 个百分点。
- **12/12 边界**：组合方案在论文 Table 1 所列滑窗/MemRoPE 比较中 12 个配置 DINOv2 均为最高；计入 Landmark/Field 消融后，组合 WorldTrace 在 11/12 配置领先，Pan 90° 是 Landmark 单独方案更高（0.917 vs 0.911）。
- **LPIPS 例外**：Pan 180° 时组合 WorldTrace 0.649，高于滑窗 0.634（反而更差）；其余 11/12 相对滑窗更低。Pan 360° DINOv2 提升仅 0.406→0.423。
- **长程 ABA N=128**：组合 0.698 DINOv2 / 0.572 LPIPS；滑窗 0.251 / 0.751。
- **跨架构**：LingBot-World 14B（附录 F.2）也有若干配置领先，但在 ABCA、ABA long、Pan 360° 等配置 MemRoPE 更好；不能推广“12/12 最佳”到全部骨干。
- **公平性/代价**：所有方法的 **attended-cache 是六帧**，但总显存并非等量；组合方案为另外的真实帧记忆库多用 **3.1 GB GPU 内存**。L40 48 GB、N=8 时滑窗 1.10 s/chunk，组合 1.40 s/chunk，耗时约 +27%。

## 局限

- 依赖具有 temporal RoPE 与已知局部注意力窗的自回归视频世界模型；其它结构不能直接套用。
- Field 聚合是有损的，轨迹越长，单槽代表的帧越多；Landmark 依赖场景进入事件检测，重要场景漏选就无法精确回忆。
- Landmark 个数上限是 `N_s`，超过后旧地标被替换；场景持久性不等于几何/动力学/动作条件真实。
- 本实验没有报告机器人真机部署或策略任务成功率。
- 项目页截至 2026-11-23 未提供可核实的官方 WorldTrace 源码仓库或运行入口；本文不包含基于真实源码的运行时序图。

## 对 wiki 的映射

- 唯一论文实体：[WorldTrace](../../wiki/entities/paper-worldtrace-addressable-memory-video-world-models.md)
- 方法导航：[Generative World Models](../../wiki/methods/generative-world-models.md)
- 评测路线：[具身模型测评 Stage 2](../../roadmap/depth-embodied-eval.md)
- 官方项目页核查：[WorldTrace 官网](../sites/worldtrace-nvidia-project.md)
