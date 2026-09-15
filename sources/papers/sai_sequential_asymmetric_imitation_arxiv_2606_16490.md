# SAI（arXiv:2606.16490）

> 来源归档（ingest）

- **标题：** 协作移动操作：序贯非对称模仿学习耦合双机策略
- **英文标题：** Robots that Collaborate: Sequential Asymmetric Imitation for Learning Coupled Robot Policies
- **类型：** paper / dual-robot / imitation-learning / mobile-manipulation / deformable
- **arXiv：** <https://arxiv.org/abs/2606.16490>（v2 2026-08-31；PDF：<https://arxiv.org/pdf/2606.16490>）
- **项目页：** <https://cyc0429.github.io/sai-project-page/>
- **机构：** 芝诺机器人（Zeno AI）；浙江大学（ZJU）；浙江工业大学（ZJUT）；悉尼大学（USYD）
- **作者：** Yincong Chen†, Ranpeng Qiu†, Zihao Li, Yanan Zhou, Guoqiang Ren, Weiming Zhi*（† 同等贡献，* 通讯）
- **平台：** 两台双臂移动操作臂（bimanual mobile manipulators）；刚体 + 可变形物体协作
- **开源：** **待发布**（步骤 2.5 核查 2026-09-15，见 [`sources/sites/sai.md`](../sites/sai.md)）
- **入库日期：** 2026-09-15

## 核心论文摘录

### 1) 问题：物理耦合下的双机协调

- 两台各自能完成局部原语的策略，在**相位失配、伙伴延迟、交互冲突**下仍会失败；失败常来自**等待/让步/拉扯/释放**时机而非局部技能差。
- **对 wiki 的映射：** [paper-sai-sequential-asymmetric-imitation](../../wiki/entities/paper-sai-sequential-asymmetric-imitation.md)

### 2) SAI 三阶段单遥操作课程

- **Stage 1：** 仅遥操作 Robot A，伙伴为顺从人类或被动伙伴 → 单边示范学基础执行。
- **Stage 2：** 冻结部署 A，遥操作 B 对抗已学 A 策略 → B 见到真实伙伴分布。
- **Stage 3：** 双机闭环部署，在协调失败点附近**稀疏干预**修正 A。
- 策略**去中心化**：不交换消息、伙伴状态、未来动作或 latent。
- **对 wiki 的映射：** 同上

### 3) 真机任务与基线

- 四类任务：床抛展被、桌布展开、洗衣收集、绘画搬运（可变形 + 刚体 + 共享工作区）。
- 对比 **Independent Imitation** 与 **Partner-Conditioned Imitation**；SAI 在任务成功率、**相位同步**、**让步/等待**指标上均提升。
- **伙伴延迟测试：** 桌布任务中暂停 B 时，独立模仿继续拉扯导致失稳；SAI 会减速、等待并在伙伴恢复后继续。
- **骨干兼容：** ACT 与 Diffusion Policy 上均优于独立模仿，说明贡献在**数据课程**而非特定动作解码器。
- **对 wiki 的映射：** 同上；交叉 [paper-trace-causal-memory](../../wiki/entities/paper-trace-causal-memory.md)、[paper-zeno-1-collaborative-intelligence](../../wiki/entities/paper-zeno-1-collaborative-intelligence.md)

## 步骤 2.5 开源核查（2026-09-15）

- 已打开 [项目页](https://cyc0429.github.io/sai-project-page/)：含摘要、任务视频、BibTeX 与 arXiv 链；**未列** GitHub / Hugging Face / 权重。
- arXiv v2 **未列** Code availability URL。
- **结论：** **待发布**；与同期 Zeno AI 研究线（TRACE / Zeno-1）作者重叠，但 SAI 代码入口仍以项目页为准。

## 当前提炼状态

- [x] 项目页步骤 2.5 核查
- [x] wiki 实体页
- [ ] 官方训练代码（待发布）
