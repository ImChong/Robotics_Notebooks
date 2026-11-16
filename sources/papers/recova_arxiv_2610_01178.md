# Recova：智能体引导的机器人操作失败恢复（arXiv:2610.01178）

> 来源归档（paper；依据论文原文，而非二手摘要）

- **英文标题：** Recova: Agent-Guided Failure Recovery for Autonomous Robotic Manipulation
- **中文标题：** Recova：智能体引导的自主机器人操作失败恢复
- **作者：** Isabella Liu, An-Chieh Cheng, Johan Bjorck, Zhiding Yu, Hongxu Yin, Jan Kautz, Linxi Fan, Yuke Zhu, Sifei Liu
- **机构：** NVIDIA、加州大学圣迭戈分校（UC San Diego）、得克萨斯大学奥斯汀分校（UT Austin）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.01178>
- **全文（HTML，v1）：** <https://arxiv.org/html/2610.01178v1>
- **PDF：** <https://arxiv.org/pdf/2610.01178>
- **项目主页：** <https://www.liuisabella.com/Recova/>
- **NVIDIA 官方镜像：** <https://research.nvidia.com/labs/gear/recova/>
- **版本：** 2026-10-01 提交，arXiv v1
- **入库日期：** 2026-11-07
- **核对范围：** 全文 Abstract、§1 Introduction、§2 Related Work、§3.1–3.4 Method、§4.1–4.2 Experiments、§5 Conclusion、附录 A（完整真机编排伪代码）、B（监控提示/判定）、C（硬件、数据、训练和评测）、D（恢复技能图例）、E（仿真失败恢复实例）；核对 Tables 1–5 与项目主页。
- **通读：** 已完成
- **一句话说明：** 将任务执行与失败恢复拆成两条策略，用编码智能体在 MuJoCo 数字孪生中发现恢复技能，并在真机部署中监测、核验、请求人工示教和分流 DAgger 数据。

## 研究问题

标准初态的成功示教难以覆盖“环卡在钉尖”“铅笔偏离托盘”等长尾失败；直接重试任务 VLA 可能无法清除障碍。Recova 不是让同一策略反复试错，而是显式学习并验证“把场景恢复到可继续操作状态”的独立能力。

## 方法原文要点

1. **任务与恢复分离**：任务策略 pi(a|o,g) 由任务指令 g 驱动；恢复策略 rho(a|o,z) 由失败对应的恢复指令 z 驱动，也可调用 code-as-policy 程序。恢复技能分为 restorative / enabling / adaptive 三类。
2. **Real → Sim**：编码智能体依据多相机画面、机器人轨迹和相机标定搭建 MuJoCo 场景，拟合物体位置与接触参数；在孪生中尝试任务、诊断失败、编写恢复技能并收集任务/恢复的成功轨迹。
3. **Sim → Real**：任务数据初始化任务策略，恢复数据初始化恢复策略，程序型恢复动作进入可复用技能库；两类监督不混合。
4. **真机监控**：VLM 通过 requirement、completion、intervention、restoration 四种查询判断任务终点、异常、是否需要恢复及恢复是否成功；恢复只需达到可继续执行的等价状态，不要求完全还原物体姿态。
5. **人工后备与 DAgger**：停滞但场景完整时采集人类任务示教；扰乱场景时优先尝试已经注册的恢复策略，否则采集人类恢复示教。连续两次介入后请求完整任务示教以避免循环；数据按 Success/Expand/Recover/Autonomous-recover 分别归入对应训练集，失败片段仅保留日志而不训练。

## 数字与实验协议

| 场景 | Recova | 对照（论文所列最强均值基线） | 口径 |
|---|---:|---:|---|
| LIBERO-Pro，六种扰动设置 | 78.8% | ASPIRE 71.7% | 六列不加权均值，每任务 50 初态；对照取先前公开结果 |
| MolmoSpaces，四类任务 | 64.9% | RATs 38.0% | Pick / Pick-and-place / Open / Close 不加权均值；对照取先前公开结果 |
| 真机四项任务：基础任务策略 | 23.8% | — | 每任务/每配置 20 次；两条 YAM 手臂 |
| 真机：基础策略 + DAgger | 77.5% | — | 与上一行同任务集合 |
| 真机：+ 部署期恢复技能 | 87.5% | — | 与上一行同任务集合 |

- 任务：装铅笔盒、叠四环、摸麻将牌、打出指定麻将牌。四工位并发采集由一名操作者监督。
- 单项“摸麻将牌”的四轮收集中，人工接管 episode 比例由 7/8（87.5%）降至 0/7（0%）；**不是全部任务、无限期无人干预的证明**。
- 真机双臂共 14 维关节/夹爪目标、30 Hz 控制；动作 chunk 50 步。微调使用 pi0.5 底座、AdamW、8× NVIDIA L40，阶段训练 2,000–5,000 步（原文 Table 5）。
- 完成查询每 600 控制步（20 秒），介入查询每 300 步（10 秒），恢复最多 1,500 步（50 秒）；附录 B 报告监控查询中位延迟约 4.5/5.0/5.5 秒（对应 intervention/completion/restoration）。

## 证据边界与复现

- 对照部分来自此前论文公开值，不能声称所有方法在统一硬件、统一实现下重跑。
- 真机 Table 3 每项仅 20 次试验，未给出全面误差条；人工介入降低的数据只覆盖一项任务的四轮采集。
- 论文与项目页展示算法伪代码、视频与部分程序型恢复脚本，但截至 2026-11-07 没发现官方完整可运行源码仓库、权重或数据集下载入口；**不能将网页内可展开的示例代码当作公开训练/部署实现**。
- 论文结尾明确将“自动注册新技能并跨任务共享”留作未来方向，不应写成已经完全自动化跨任务共享。

## 对 wiki 的映射

- [Recova 统一论文项目节点](../../wiki/entities/paper-recova.md)
- [Manipulation 操作任务](../../wiki/tasks/manipulation.md)

## 2026-11-08 补充核验：附录伪代码、逐项结果与可复现边界

- **阅读范围：** 再次核对官方 [arXiv HTML v1](https://arxiv.org/html/2610.01178v1) 的 §1–5、Tables 1–5、Figures 1–10、Appendix A–E，交叉对照 [NVIDIA 官方项目页](https://research.nvidia.com/labs/gear/recova/)；论文 PDF 共 23 页。原文通读范围仍为已发布正文和全部附录，未独立执行训练或真机实验。
- **真机四任务逐项结果（Table 3，单位 %，每格 20 trials）：**

| 配置 | Pencil box | Stack rings | Draw tile | Discard tile | Avg. |
|---|---:|---:|---:|---:|---:|
| Base | 20.0 | 35.0 | 25.0 | 15.0 | 23.8 |
| + DAgger | 70.0 | 75.0 | 85.0 | 80.0 | 77.5 |
| + Recovery | 85.0 | 90.0 | 90.0 | 85.0 | 87.5 |

- **独立统计口径（Fig. 4）：** 摸牌任务在第一轮人类接管 7/8 次，第四轮 0/7 次；任务策略自主完成从 1/8 增至 6/7。另一次 14 分钟四站同时采集的案例中，策略完成 22/31 个采集 episode，人工控制占累计站点时间 11%，两组统计不能混同。
- **仿真与真机不能混称：** §4.1 仿真基准采用 **MolmoAct2 task policy + programmatic recoveries**；§4.2 真机采用两条分别微调的 **π0.5 task / recovery policies**，而非单一端到端策略。
- **动作语义（C.3、Table 5）：** 14 维动作中 12 个手臂关节使用以当前状态为基准的 offset，2 个夹爪命令是 absolute；50 步 action chunk @30 Hz；图像输入 top + 两腕相机 224×224；四视角 640×480 采集给监视器。
- **编排伪代码（Appendix A Algorithm 1）：** 任务策略每 600 步检查完成（机器人保持静止），每 300 步异步发起干预判断；已注册恢复技能单次最多 1500 步并经 restoration 检查；任务尝试上限 5400 步；连续两次干预请求完整人工任务示教。监视器四种查询：requirement、completion、intervention、restoration；判断延迟中位数分别约 5.0 s（completion）、4.5 s（intervention）、5.5 s（restoration），对应附录 B。
- **异步状态处理（Appendix A 伪代码后说明）：** 弃用结束轮次或人工改动场景前的过期查询结果；干预结果先于在途 completion 到达时必须暂停并等待后者，完成优先；不应把慢 VLM 当作 30 Hz 的低层安全控制。
- **训练分流（§3.4）：** `S_k` 自主成功与 `E_k` 人类任务示教更新任务策略；`R_k` 人类恢复示教与 `A_k` 核验通过的自主恢复更新恢复策略。失败前缀日志留存但不纳入监督目标。
- **正文/附录数目差异：** §4.2 Setup 写总计 25 recovery skills，附录 D 明示 Figures 8–9 共展示 24 种。保留原文原状，不自行补齐缺失技能。
- **源码与复现：** 论文给出 Algorithm 1 *伪代码*，官方网站可查看一些恢复技能演示/片段；这不足以认定训练与部署源代码已经完整发布。新增的知识页时序图仅对应论文伪代码，明确**不是源码时序图**。

证据链接：[HTML §3–4 与 Appendix A–E](https://arxiv.org/html/2610.01178v1)、[PDF](https://arxiv.org/pdf/2610.01178)、[NVIDIA 官网](https://research.nvidia.com/labs/gear/recova/)。
