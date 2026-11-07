# Δ Flow Matching for Robot Foundation Models / Δ 流匹配：从历史动作预测未来变化

> 来源归档（paper）｜一手论文复核

- **标题：** Δ Flow Matching for Robot Foundation Models
- **机构：** 南洋理工大学 MARS Lab（Nanyang Technological University）
- **作者：** Kuangji Zuo、Tuo An 等 12 人
- **类型：** paper
- **版本：** arXiv:2610.15794v1，2026-10-11
- **arXiv：** <https://arxiv.org/abs/2610.15794>
- **全文 HTML：** <https://arxiv.org/html/2610.15794v1>
- **PDF：** <https://arxiv.org/pdf/2610.15794>
- **项目页：** <https://zuo-kuangji.github.io/delta-flow/>
- **官方仓库：** <https://github.com/NTUMARS/delta-flow-matching>
- **首次入库：** 2026-10-14
- **本次一手核验：** 2026-11-07
- **核对范围：** 通读论文 v1 正文 §1–9（相关工作、预备知识、方法、训练、评测、消融、设计研究、局限/未来工作）和附录 A–E；逐项核对 Eq.(3)–(9)、Table 1–9、图示及项目页；参考文献按需查阅。
- **通读：** 已完成
- **一句话说明：** 将流匹配的源分布从独立高斯噪声替换为轻噪声扰动的执行历史，以较短残差运输实现动作与视觉潜变量的一步预测。

## 核心摘录（直接依据论文，非转载）

1. **来源构造（§4，Eq.3–5）：** 对动作或视觉模态 m，采用 s_m = α_m h_m + σ_m (b_m ⊙ ε_m)。h_m 是同维度历史，b_m 是噪声掩码；训练沿 (1−τ)y_m + τs_m 插值，学习速度场 s_m − y_m，并增加一步 Euler 终点的一致性 L1 损失。
2. **适配边界（§4.1–4.3）：** π0.5 只修改动作分支的源和目标；ImageWAM / Fast-WAM 对视觉与动作两分支均使用历史源。视觉中间特征用于条件化动作分支，动作梯度在该连接处停止。
3. **训练效率（§5.2）：** ImageWAM 在 LIBERO 达到至少 90% 的 epoch 从 6 变为 1（首 epoch 91.30%）；Fast-WAM 达到至少 95% 从 8 变为 2（第 2 epoch 95.65%）；π0.5 的 LoRA 第 1 epoch 92.20%，超过高斯十步基线第 5 epoch 的 91.30%。这些是达到阈值的 epoch 比较，不等价于端到端墙钟训练提速比例。
4. **完整策略调用延迟（§5.4；附录 B.3）：** 在单张 A100-SXM4-80GB、batch=1 下，π0.5 为 82.91→53.99 ms（1.54×）；ImageWAM 为 296.31→118.87 ms（2.49×）；Fast-WAM sparse joint 为 857.54→192.62 ms（4.45×）。均与十步高斯参考设置比较；排除仿真步进、网络通信等。
5. **泛化与平滑（Table 1、§5.4）：** ImageWAM 的 LIBERO-Plus 总成功率 82.49→88.55%；相应标准 LIBERO 最终值为 98.00→97.80%，非所有指标都单调提升。在 LIBERO-Long 的相匹配成功轨迹上，完整 Δ-flow 相比一步高斯的平均 jerk 约低 4.5 倍。
6. **无重训替换（§5.5，Table 2）：** 冻结已按高斯训练的 ImageWAM / Fast-WAM，只在推理时替换动作源。LIBERO 一步成功率分别 97.50→98.10% 与 95.55→96.15%；Fast-WAM 在 LIBERO-Plus 却从 68.28% 降为 67.58%，不是普适无损替代。
7. **设计结论与局限（§6–9）：** 视觉 Δ-flow、视觉终点一致性、stop-gradient 与局部增量动作表征都影响泛化；附录给出 8×A100 训练设置。作者明确把更多 VLA 基座与真实机器人验证列为未来工作。

## 开放状态（2026-11-07 核验）

**论文和项目说明公开；官方训练/评测代码、模型权重与可执行复现入口尚未发布。** 不能把有 GitHub README 误记为代码已开源。项目页的交互示意不是机器人本机复现实验。

## 对 wiki 的映射

- [Δ-flow 主实体](../../wiki/entities/paper-delta-flow-matching-robot-foundation-models.md)
- [2026-10-14 论文技术地图](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
