# Σ-LeWM（arXiv:2611.12732）

> 来源归档（paper）

- **标题：** Σ-LeWM: Learning World Models via S-Information for Robot Control
- **类型：** paper
- **作者 / 机构：** Xinrui Zu, Michael Przystupa, Zhao Yang, Shujian Yu, Kevin Sebastian Luck；Vrije Universiteit Amsterdam（荷兰）
- **arXiv：** <https://arxiv.org/abs/2611.12732>（v1，cs.LG，2026-11-09）
- **PDF：** <https://arxiv.org/pdf/2611.12732v1>
- **项目页：** <https://zuxinrui.github.io/sigma-lewm/>
- **代码：** https://github.com/zuxinrui/sigma-lewm
- **入库日期：** 2026-11-11
- **一句话说明：** 在 LeWorldModel（解码器-free 潜在世界模型）上加反向动力学与逆动力学两项对比式辅助损失（DCL），按 S-Information 解释，推理时仍只用前向模型做 CEM 规划；全部实验在仿真中。
- **核对范围：** 下载 arXiv v1 PDF，用 pdftotext 通读第 1–6 节正文（含公式 1–20、表 1–2）、附录 A（线性探针，表 3）与附录 B（规划延迟，表 4）和参考文献列表；图 2–9 的数值取自 PDF 文字层与渲染页，仅渲染查看了第 10 页（图 6），其余图以文字层标注为准；另静态读了项目页文字与仓库 README（raw 文件）。未读：仓库源码与 docs/ 子文档、HF 检查点；未运行任何代码；未核对被引文献原文。
- **通读：** 已完成

## 开源状态

- 仓库 `zuxinrui/sigma-lewm` README 声明「仅含代码与配置，不含数据集、检查点、日志与实测数字」；四个标准任务的检查点托管在 Hugging Face（`zuxinrui/sigma-lewm`）；Gear / Franka 需自行安装 Isaac Lab 与任务资产（未随库分发）。核查日期 2026-11-11（只读 README，未验证 HF 内容）。

## 核心摘录

1. **方法（第 4 节）：** 编码器 E、动作编码器 E^A、前向动力学 f 保留；新增反向动力学 f_back(z_{t+1}, ã_t)→z_t 与逆动力学 f_inv(z_t, z_{t+1})→ã_t，均用 decoupled contrastive loss（DCL，式 5–6）训练，权重 λ_back=0.1、λ_inv=0.05（第 5.1 节）；另加多步前向项（λ_multi=1，最大偏移 m=5）。辅助头仅训练期使用，测试期规划与 LeWM 相同（第 4.1 节）。
2. **理论（第 4.3 节）：** 前向、反向、逆三项互信息之和等于 TC + DTC，即 S-Information（式 19–20）；这是对目标的信息论解读，论文没有证明下游控制性能与之成正比。
3. **标准基准（表 1，h=5，CEM30，3 训练种子×5 评测种子×50 回合）：** Push-T 94.5 vs LeWM 90.0；OGBench-Scene 80.7 vs 57.7；Cube 82.6 vs 66.5；TwoRoom 92.4 vs 90.9；Reacher 86.0 vs 86.0。第 5.2 节自述 TwoRoom、Cube、Reacher「在噪声范围内」。
4. **规划效率（第 5.4 节）：** Push-T 上 CEM8 达 91.2%，高于 LeWM 在 CEM30 的 85.2%。附录 B 表 4：RTX 3090 上 CEM30→CEM4 求解延迟 692.2→99.5 ms，实测加速约 6.9×（H100 为 6.5×）。
5. **仿真机器人任务（第 5.5 节，图 6）：** IsaacSim 中 Gear-Assembly 严格成功率 LeWM 10.9% → Σ-LeWM 74.9%；同图中 AC-MTM 65.6、SMWM 64.9（取自图 6 文字层）。Cube-Stacking 为图 6 柱状图数字（Σ-LeWM 36.8、SMWM 30.2、AC-MTM 20.2、LeWM 4.0），正文只对 Gear 给出文字结论。
6. **消融（第 5.7 节，图 9，表 2）：** OGBench-Cube h=5：LeWM 64、base（仅多步）71、+逆 72、+反向 80、两者 84（项目页同数）；回归（MSE）版本不如对比（DCL）版本。损失权重在 Push-T 上扫 0.5×/1×/2×，默认值最好（表 2）。
7. **作者承认的范围（第 6 节）：** 未来工作才包括「在规划时使用辅助任务」与「在真实机器人任务上部署」。

## 对 wiki 的映射

- [paper-sigma-lewm](../../wiki/entities/paper-sigma-lewm.md)
- 前作 LeWM：[lewm_arxiv_2603_19312](lewm_arxiv_2603_19312.md)
