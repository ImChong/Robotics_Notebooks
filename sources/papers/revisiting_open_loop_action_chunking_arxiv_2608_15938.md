# Revisiting Open-Loop Execution in Robotics（arXiv:2608.15938）

> 来源归档（ingest）

- **标题：** Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies
- **类型：** paper / imitation-learning / action-chunking / analysis
- **arXiv：** <https://arxiv.org/abs/2608.15938>
- **项目页：** <https://revisiting-open-loop-action-chunking.github.io/>
- **机构：** 麻省理工学院（MIT）；加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-08-22（全文通读与复核 2026-11-19）
- **通读：** 已完成
- **核对范围：** 通读 arXiv:2608.15938v2（2026-08-19）正文与 Appendix A–I 全文，核对 Table 1 真机 ARX-5 结果、FurnitureSimOneLeg 千条演示与 HG-DAgger 实验、App F 双编码器（double encoder）与 App I checkpoint 规律
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_open_source_2026-08-23.md)（<https://mp.weixin.qq.com/s/CXOf3PU8-H6OzI77vnhZMA>）
- **一句话说明：** 系统实验表明：长 **open-loop execution horizon** 主因是 **短上下文策略模仿非马尔可夫专家**；足够长的观测上下文下 **闭环 reactive 策略最优**；提出 **double encoder** 稳定长上下文 Diffusion Policy 训练。

## 开源状态（步骤 2.5，复核 2026-11-19）

| 资源 | 状态 |
|------|------|
| 项目页 / arXiv（v2） | **已发布** |
| GitHub 总索引仓 | **已挂链但仅含 README** — [`Michaelszeng/Revisiting-Open-Loop-Execution-in-Robotics`](../repos/revisiting-open-loop-execution-in-robotics.md)（仅 `LICENSE` 与 `README.md`，索引 6 个子仓 + 2 个 HF 数据集；**不含可运行主策略源码**） |
| Section 4 / 真机推理源码 | 仿真子任务分散于 6 个外链子仓；真机 ARX-5 推理栈属团队内部基础设施**未开源** |

**结论：** **部分开源（索引仓 + 子组件仓）/ 主仓无可运行代码** — 机制与源码归档见 [`sources/repos/revisiting-open-loop-execution-in-robotics.md`](../repos/revisiting-open-loop-execution-in-robotics.md)。

## 核心论文摘录

### 1) 核心论断（Abstract）

- **核心贡献：** Action chunking 的 **open-loop 执行前缀**（execution horizon \(T_{\mathrm{exec}}\)）降低反应性。本文认为：在 **短 context**（常见 \(T_o=1\)–\(2\)）下，长 \(T_{\mathrm{exec}}\) 主要补偿 **专家演示的非马尔可夫性**（隐式记忆、暂停、模式承诺），而非单独由复合误差或推理延迟解释；**复合误差有影响，但弱于非马尔可夫性**。
- **对 wiki 的映射：**
  - [Revisiting Open-Loop 论文实体](../../wiki/entities/paper-revisiting-open-loop-action-chunking.md)
  - [Action Chunking](../../wiki/methods/action-chunking.md)
  - [Why Action Chunking Improves BC](../../wiki/entities/paper-why-action-chunking-improves-bc.md)（并发机制对照）

### 2) 实验设计（§3–4）

- **变量：** 固定 prediction horizon \(T_p\)，扫 \(T_{\mathrm{exec}}\) 与 \(T_o\)（context length）；success-horizon 曲线。
- **任务：** FurnitureSimOneLeg、GearInsertion、PushT-D、Kitchen（仿真）；SinglePillDispense、SlipIntoBaggie（双 ARX-5 真机）。
- **干预：** HG-DAgger 测试复合误差假说；Markov vs 非 Markov 专家对照。
- **对 wiki 的映射：**
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
  - [Behavior Cloning](../../wiki/methods/behavior-cloning.md)

### 3) 长上下文与 double encoder（§5 / Appendix F）

- **发现：** 增大 \(T_o\)（如 8–20）逐步消除长 \(T_{\mathrm{exec}}\) 收益；数据充足时长上下文 **reactive** 策略优于短上下文长执行。
- **方法：** **Double encoder** — 短程 \(E_S\) 与长程 \(E_L\) 分离最近帧与历史；短程 dropout 防忽略长程特征。
- **对 wiki 的映射：**
  - [LIBERO benchmark](../../wiki/entities/libero-benchmark.md)（对照生态）

### 4) 与 Why AC Improves BC 的分界

- **Revisiting（本文）：** 聚焦 **execution horizon** vs **context length**；主张足够上下文下 **不必长 open-loop 执行**。
- **Why AC（CoRL 2026）：** 聚焦 **chunk 训练目标** vs **Delay/RDE 部署**；强调隐式集成与延迟条件化。
- **对 wiki 的映射：**
  - 两文互补写入 [action-chunking](../../wiki/methods/action-chunking.md) 机制节
