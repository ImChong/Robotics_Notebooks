# Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning（arXiv:2611.08144）

> 来源归档（paper）

- **标题：** Detecting Silent Faults at the Environment–Trainer Boundary in Reinforcement Learning
- **作者 / 机构：** Zuo Gou、Chenhao Lu（共同一作），清华大学（论文首页）
- **类型：** paper（arXiv v1，cs.RO，2026-11-05）
- **arXiv：** <https://arxiv.org/abs/2611.08144>
- **PDF：** <https://arxiv.org/pdf/2611.08144v1>
- **代码 / 资源：** 论文首页与 Sec. I 给出 <https://github.com/PlugRL>（含实验协议、日志、规范与检查器）；本次未能访问该组织页（代理返回 403），内容未核实
- **入库日期：** 2026-11-09
- **来源路径：** 具身智能小站 2026-11-09 更新（29 篇论文中的扫读条目），小站条目「训练曲线正常，也可能传错了数据」
- **一句话说明：** 把环境与训练器之间的边界做成显式线协议（PlugRL），向其中注入静默错误，发现学习曲线与干净评估基本看不出来，而两侧各留逐步记录并对账（step ledger）能全部发现并定位。

## 核对范围与通读状态

- **核对范围：** 读了 arXiv v1 PDF 全部正文（Sec. I–IX）、致谢、参考文献列表与附录 A–G（含 Fig. 1–6、Table I–VI）；图表通过 pdftotext 抽出的文字、坐标标注与图注核对，未逐幅查看位图。未读：论文引用的 GitHub 仓库 PlugRL（实验协议 E43–E58、规范与检查器代码，访问失败）、各引用文献原文。因此「实验编号对应的原始日志是否与文中数字一致」未复核。
- **通读：** 已完成

## 论文要点（带章节）

1. **问题设定（Sec. I、II）：** RL 环境越来越多地跑在训练器之外（另一进程、机器、语言）。作者把边界写成 WebSocket + msgpack 的协议（infer / action / feedback，加 stop / resync），并配规范与双侧一致性检查器；Stable-Baselines3、CleanRL、RLinf 通过适配器接入。
2. **故障注入（Sec. III）：** 用 SB3 的 PPO 在 Pendulum、HalfCheetah、Hopper、Pusher（7 关节臂）上，在环境客户端对观测、动作、奖励、终止观测、回合结束、步骤等字段做置零、取上一步值、换成下一环境值、float16 取整、动作裁剪、虚构或漏掉回合结束、跳步填充等；另有 2 种写入训练器日志的故障。剂量为 1 个值、0.1%、1%、10%、全部值；种子 0–2。
3. **曲线类检查基本看不出（Sec. III、Fig. 2/3）：** 剂量不超过 10% 时，增益阈值（Pendulum 上要求训练末期比起点回报高 500）在四个环境上最多抓到十分之一的运行；全部值出错时抓到 43%、35%、48%、33%。种子离散度检查（离十个无故障种子均值超过 3 个标准差）在低于全量的剂量下最多抓到 21%。对 1,060 个最终策略做干净环境评估（每个 50 个确定性回合）：两项曲线检查都通过的运行里，Pendulum 255 个中有 5 个、其余三个环境 473 个中无一个低于无故障种子 3 个标准差。
4. **逐步账本全部发现（Sec. III、Fig. 2）：** 环境侧包装器与向量环境侧包装器各记每步观测、动作与返回值，逐步对账：Pendulum、Hopper、Pusher 的 189、207、189 个边界故障运行全部发现并定位到首个出错的步与字段，27 个无故障运行均无差异；在 Pendulum 上训练时间中位数增加 3%（范围 -5% 到 +9%），权重不变；笔记本加工作站两台机器上 21 个单值故障全部发现。HalfCheetah 未跑账本。
5. **回合记录与字节一致（Sec. III）：** 环境端回合记录抓到全部日志故障与改变奖励（超出容差）、回合结束、步骤的边界故障，但抓不到观测、动作、终止观测、回合结束类型错误、跳步；训练器侧的 bridge log 只抓到日志故障。字节一致（同种子同机器权重逐位相同）在 Hopper 漏掉 12 个只改了终止观测的运行（SB3 只在时间上限时从终止观测自举，训练未读该值）。
6. **实际案例（Sec. IV、Table I、Appendix G/Table VI）：** Gymnasium 1.3 在 SAME_STEP 自动重置下，向量环境回合统计丢掉除首个回合外每回合的第一个奖励且长度少计 1，1.4 修复；CleanRL 在 Gymnasium 1.3 上不记录回合或报 TypeError；另有 RLlib、Tianshou、PettingZoo、SuperSuit、MAPPO 的 SMACv2 封装等他人报告的同类问题（Table I，作者引用 issue 或 PR 而非自己复现）。
7. **边界移动与代价（Sec. V、VI、Table III–V、Appendix C/D）：** 单机上进程内与经桥运行权重逐位一致；Pendulum 在 0–25 ms 延迟代理后仍一致；桥每个向量步多 0.4–0.6 ms（16 个状态向量）或 2.6–3.0 ms（16 帧）；跨机器后种子不再复现（两台机器 MuJoCo 构建首步相差 8.9e-16），退而用增益阈值，各机器各种子均通过。
8. **局限（Sec. IX）：** 故障由作者选择，仅四个仿真、状态输入、稠密奖励环境，仅 SB3 PPO、3 个种子；稀疏奖励、长时域、回放缓冲、图像、真机上结论未必成立；账本需要包装环境，看不到环境内部的错误；账本事后比对（Hopper 一次 150 MB）；规范与 C++ 客户端同作者；不研究环境在等动作时仍在运动的真机情形。

## 对 wiki 的映射

- [paper-silent-faults-env-trainer-boundary](../../wiki/entities/paper-silent-faults-env-trainer-boundary.md)
