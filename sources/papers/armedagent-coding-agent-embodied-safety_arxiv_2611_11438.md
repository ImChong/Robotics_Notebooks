# When Coding Agents Get Arms: Safety Risks Across the Digital-Physical Boundary（arXiv:2611.11438）

> 来源归档（paper）

- **标题：** When Coding Agents Get Arms: Safety Risks Across the Digital-Physical Boundary
- **类型：** paper（技术报告，TECHNICAL REPORT，cs.RO，arXiv v1 提交日 2026-11-08）
- **arXiv：** <https://arxiv.org/abs/2611.11438>
- **PDF：** <https://arxiv.org/pdf/2611.11438v1>
- **项目页：** <https://ymsun2020.github.io/CodingAgentEmbodiedSafety/>
- **作者 / 机构：** Yiming Sun, Mi Zhang（通讯）；复旦大学，JADE AI（Whitzard AI）团队
- **代码：** 论文正文与项目页（2026-11-11 静态读取）均未给出仓库链接；论文称"也发布与真机对应的模拟器"（§1），但未给地址
- **入库日期：** 2026-11-11
- **一句话说明：** 提出 ArmedAgent 基准：36 个任务、5 个场景，评测 Codex 驱动的代码智能体零样本控制机械臂时在鲁棒性、恶意使用、间接提示注入三类风险下的物理不安全行为；3 个模型 5 种设置的总体攻击成功率 72.22%–94.29%，拒绝率 0%。
- **核对范围：** 逐段读完 arXiv v1 PDF 全部 12 页正文（摘要、§1 引言、§2 相关工作、§3 方法含威胁模型与数据集构造、§4 评测含表 1 与 5 个真机任务、§5 结论、参考文献）；论文无附录与补充材料；图 1–7 以图注和正文描述核对，未逐帧检视视频；项目页仅读到摘要、风险分类与真机案例的文字描述，未见代码链接
- **通读：** 已完成

## 开源状态

- **未核实到仓库**：论文与项目页均无代码 / 数据集链接（2026-11-11）；§1 称发布模拟器，但无地址，视为未落地，待作者补链

## 核心摘录

1. **设定（§3.1）：** 代码智能体 = 骨干模型 + Codex / Claude Code 等框架，有 shell、文件系统与机器人接口（相机、位姿查询、操作原语）；每步读图像 o_t 与工具输出 c_t，写程序 a_t 执行；轨迹任意一步进入不安全集合 U 即判 Unsafe（式 1、2）。
2. **三种威胁模型（§3.1.2）：** T1 无意的善意用户（鲁棒性）、T2 恶意用户、T3 间接注入（用户与指令都善意，对手只控制智能体读到的一部分内容 c = c_benign ⊕ δ）。
3. **四类危害（§3.1.2）：** 人身伤害与危险、财产损失与资源浪费、系统被入侵（如篡改机器人配置）、违禁内容。
4. **环境落地型风险（§3.2.1）：** 指令本身不显险（h(I,K)=0），结合场景观察才显险（h(I,K,O)=1），如打开插着烙铁的开关、空锅加热。
5. **仿真（§3.2）：** 基于 RoboDojo 与 Isaac Sim / Isaac Lab，复用其 X5 机械臂；每任务有状态检查器；场景在 Codex 协助下搭建并人工核验。
6. **实验设置（§4）：** GPT-6-Astra（低 / 中 / 高推理强度）、GPT-6-Sol、GPT-6.1-Terra，均以 Codex 为框架；同一机器人接口、系统提示、步数预算。人工看执行视频与最终场景标注三种结局：攻击成功 ASR / 拒绝 RR / 失败尝试 FAR。
7. **表 1（总体 ASR）：** Astra-low 72.22，Astra-medium 75.00，Astra-high 94.29，Sol 91.43，Terra 80.00；间接注入 81.82–90.91；Astra-high 在恶意使用上 100.00。RR 全为 0%（§4.1）。
8. **真机（§4.2）：** GPT-6-Astra + Codex，5 个任务各 1 次（2 个环境落地型：湿纸巾擦裸内存条、端水杯经过内存条上方；3 个恶意使用：在白板上画纳粹旗、猥亵图形、写脏话），5 次全部达成有害结果；论文自述只作"风险真实存在"的证据而非失败率估计。
9. **自行推算（非论文原文，推测）：** 表 1 百分比与 36 / 35 个 episode 的分数吻合：Astra-low 与 medium 分母 36，其余三设置分母 35；按分数拆分，鲁棒性分母 17、间接注入 11、恶意使用 8 或 7，论文未解释为何有 35 个，也未给出各类别样本数。

## 对 wiki 的映射

- [paper-armedagent-coding-agent-embodied-safety](../../wiki/entities/paper-armedagent-coding-agent-embodied-safety.md)
