# FlowDPG（CoRL 2026）原文通读与核验

## 论文身份
- **英文标题：** FlowDPG: Deterministic Policy Gradient on Flow Matching Policies for Real-World Manipulation
- **中文标题：** FlowDPG：面向真实机器人操作的流匹配确定性策略梯度
- **作者：** Kexin Shi、Junyao Shi、Poorvi Hebbar、Zhuolun Zhao、Tarun Amarnath、Yifan Su、Shikhar Bahl、Deepak Pathak
- **机构：** Skild AI、Carnegie Mellon University、University of Pennsylvania
- **收录：** CoRL 2026；项目页亦列 RSS 2026 Diff4RL Spotlight
- **arXiv：** 2606.22303v2（2026-09-30，camera-ready）
- **全文 HTML：** https://arxiv.org/html/2606.22303v2
- **官方 PDF：** https://flowdpg.github.io/static/images/main.pdf（17 页）
- **项目主页：** https://flowdpg.github.io/
- **入库日期：** 2026-11-13
- **核对范围：** 通读 v2 Abstract、§1 Introduction、§2 Related Work、§3 Method（公式 1–11）、§4 Experiments（Tables 1–3、消融）、§5 Conclusion/Limitations、Appendix A 敏感度推导、Appendix B Algorithm 1、Appendix C 奖励/价值曲线、Appendix D 两项任务评分量表；对照官方 PDF 中的 Fig. 1/3 与项目页全部研究结果及资源入口。
- **通读：** 已完成

## 核心机制
1. **问题：** 对多步 Flow Matching ODE 的终端动作直接回传 Critic 梯度，需要长链 BPTT，算力与梯度稳定性较差（§1、3）。
2. **估计价值：** twin-critic IQL，以 expectile 状态价值和 min(Q1,Q2) 避免在未见动作上做显式 max；阶段感知奖励模型 SARM 输出进度、阶段转换、终局三类回报（公式 1–3）。
3. **单步投影：** â = x_t + (1−t)vθ(x_t,t,s)；在 â 处计算 g=∇â min_i Q_i(s,â)，停止 Critic 参数梯度（公式 4–5）。
4. **梯度蒸馏：** Δ=α||u_t||g/(||g||+ε)，a*=â+Δ，u*=a*−x0；以 L2 目标回归速度场并保留 BC anchor；λ 从零 warmup，一致性损失 ||â−x1||² 保证清洁动作估计可靠（公式 6–9）。
5. **理论边界：** FlowDPG 是 vanilla DPG 的三个近似：单步估计替代 ODE 末端并近似敏感度、单点 Monte Carlo 替代沿时间积分、用动作速度尺度归一化梯度幅值；不是严格等价（§3.2、Appendix A）。
6. **离线→在线：** 演示数据训练 backbone、actor、value；真机 rollout 异步写入在线缓冲区，按约 1:1 离线/在线采样，在线阶段冻结 backbone 与 SARM，BC 锚点仅示范与成功 rollout，定期同步策略参数。推理仍采用多步 ODE，无部署 Critic 查询（§3.3、Algorithm 1）。

## 实验与口径
| 项目 | 双 Franka 组装 AirPods | 双 YAM 炒蛋 |
|---|---|---|
| 阶段 | 8：抓盒/开盒/抓插左右耳机/关盒/放盒 | 14：两次取蛋打蛋、加盐、取叉搅蛋、入锅、取铲翻炒、盛盘、归位 |
| 图像 | 4 路：前、顶、双腕 | 3 路：顶、双腕 |
| 示范 | GELLO 遥操作 | GELLO 遥操作 |
| 结果 | BC 64%、QAM 离线 80%、FlowDPG 离线 88% / 在线 92% 整任务成功率 | BC 64.2%、FlowDPG 离线 85.8% 的 14 阶段 rubric |

- AirPods 主对照为每种方法 25 个完整 case；整体成功要求两枚耳机均正确装入，阶段内允许失败重试。另有跨 5 个 checkpoint 各 10 次、总 50 rollout 的重复评估，在线 FlowDPG 整任务率 90%；协议不同不应混用。
- 炒蛋报告的是每阶段 0–3 分、14 阶段满分 42 分的 rubric 百分比；约 86% **不是整任务成功率**，且炒蛋只测试离线模型。
- Table 2 人为扰动恢复（50 次/类，离线→在线）：物体重新放回 78→92%；盖子状态翻转 82→94%；耳机脱位 86→92%；托盘在伸手过程中重排 70→88%。
- Fig. 5–6：无一致性正则成功率 92→80%，无自适应位移 92→76%，仅终局奖励末阶段覆盖率降至 60%。
- 项目主页的一小时未剪辑 25-case AirPods 视频是定性演示，不替代上述定量评测。

## 局限和源码
- Critic 在分布外可能错误评价，SARM 对新状态的进度/阶段判断不准时在线更新也会失真；任务需可划分阶段的密集奖励，评测只覆盖两项任务及两类双臂平台；大规模 VLA 扩展仍待研究（§5）。
- 论文仅称 two Franka arms，未独立确认商品型号 Franka Duo；炒蛋由 YAM 双臂完成。商业广告提及的 PNP/Manus/Wuji 不属于本文的实验支持结论。
- 截至 2026-11-13，项目主页列 Paper/arXiv/Video/BibTeX，但**未列可核实的官方源码、权重或数据集下载入口**。论文 Algorithm 1 是伪代码，不能假装是代码运行时序。

## 对 wiki 的映射
- [FlowDPG 独立实体](../../wiki/entities/paper-flowdpg.md)
- [双臂操作](../../wiki/tasks/bimanual-manipulation.md)
