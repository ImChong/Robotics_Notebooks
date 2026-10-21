# PACER: Process-Aware VLA Policy Learning from Human Corrections for Robotic Disassembly（arXiv:2610.19248）

> 来源归档（ingest）

- **标题：** PACER: Process-Aware VLA Policy Learning from Human Corrections for Robotic Disassembly
- **arXiv：** <https://arxiv.org/abs/2610.19248>（PDF：<https://arxiv.org/pdf/2610.19248>）
- **代码（官方仓库，已核查）：** <https://github.com/ChangChrisLiu/PACER> — 归档见 [`sources/repos/pacer.md`](../repos/pacer.md)
- **项目页：** <https://pacer-corl.github.io/> — 归档见 [`sources/sites/pacer-project.md`](../sites/pacer-project.md)
- **作者：** Chang Liu, Sibo Tian, Litian Gong, Zhiwen Fan, Jiachen Li, Xiao Liang, Minghui Zheng（通讯）
- **机构（arXiv HTML 版）：** Texas A&M University（Liu、Tian、Fan、Liang、Zheng）；UC Riverside（Gong）；Georgia Institute of Technology（Li）
- **入库日期：** 2026-10-17
- **一句话说明：** 在 VLA 的 SFT 之上，用过程证据（进展、终端区域、停止/交接一致性、操作员标签）对 rollout 与人工纠正的动作块重加权，保持原生 flow 回归损失不变；UR5e 主板拆解 approach-and-handoff 87/100。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 页面说明论文标注为 "soon"，代码以 MIT 协议开源，链接 `github.com/ChangChrisLiu/PACER`。
- **仓库核查（2026-10-17）：** README 称这是 "research framework and runtime-adapter reference"，含代码、schema、合成示例、测试与安装说明；**不含**私有机器人日志、视频、检查点、凭据、结果表。MIT 许可。
- **结论：** **部分开源**（通用框架与运行时适配层可无硬件跑 mock / dry-run；真实数据、权重、机器人驱动未随附）。本次仅阅读 README，未本地运行。

## 摘录 1：方法（arXiv HTML）

- 资格门：仅在训练划分内、掩码与来源有效、安全执行且非失败/排除的行可提供训练目标。
- 过程证据：TCP 进展、终点接近度、终端区域进入、方向（仅验证用）、停止/交接一致性、操作员结果（clean 与 correction=1.0，auto-success=0.8，partial=0.6）、来源。
- 角色乘数：clean / correction / auto-success / partial / failure / excluded；clean 与 correction 有保护下限 0.80 与 0.65；权重有上限裁剪。
- 损失：对原生 flow 回归损失做 batch 归一化的逐 horizon 加权，权重按行共享。
- 选择：8 个预先声明的加权设定各训练一个 LoRA 适配器，按分量均衡的离线验证分数（且通过审计）选一个。

**对 wiki 的映射：** [`paper-pacer.md`](../../wiki/entities/paper-pacer.md) 核心原理；与 [DAgger](../../wiki/methods/dagger.md)、[HIL 混合模仿](../../wiki/methods/hil-hybrid-imitation-learning.md) 对照。

## 摘录 2：实验设置与结果

- UR5e，主板五类部件：CPU、GPU、CPU 风扇、RAM、连接器块；基座 π0.5-DROID，在 710 条遥操作轨迹上 SFT；每部件 70 条自主 rollout（成功数 4、31、11、7、2）加纠正与补充 clean 演示；每部件 60 训练 / 10 验证配置组。
- 批 32，每个策略 10,000 步；测试为 100 次留出硬件试验（每部件 20 次，两种场景配置）。
- Table 1（approach-and-handoff 成功 / 100）：Initial SFT 12；Vanilla post-SFT 54；Uniform all-eligible 64；Outcome only 11；Clean only 24；Correction only 0；Fixed geometry 79；PACER 87（95% Wilson CI 79.0–92.2%）。分部件：RAM 19、connector 13、CPU 15、GPU 20、CPU fan 20。
- Table 2（受控 OOD，100 次/策略）：PACER 72；Fixed geometry 51；RW-like 51；Uniform 48；Role only 39；No clean 35；No correction 13；No outcome 50。

**对 wiki 的映射：** 实验与评测、结论各条数字。

## 摘录 3：作者自述局限

- 依赖准确的轨迹元数据（目标几何、角色、标签、掩码、来源）；更广泛的元数据错误未测。
- 顶部验证候选在重采样下难以区分，验证分数与硬件表现仅部分校准。
- 消融未隔离单个证据特征、角色乘数、下限或选择规则的作用。
- 仅一个任务域、一台机器人、一个 VLA；未在标准操作基准上测试。

**对 wiki 的映射：** 局限与风险。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-pacer.md`；仅新增实体页。
