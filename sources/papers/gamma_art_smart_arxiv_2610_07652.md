# SMART / Gamma-Art：铰接物体大规模合成预训练论文归档

- **标题：** SMART: Zero-Shot Sim-to-Real Articulated Object Manipulation via Large-Scale Synthetic Pretraining
- **论文：** https://arxiv.org/abs/2610.07652（2026-10-06，v1）
- **HTML 全文：** https://arxiv.org/html/2610.07652v1
- **作者：** Jicong Ao、Shuhan Jiang、Yuling Zhong、Yanwen Liu、Yuhan Gao、Jiangyuan Zhao、Yang Zhang、Shiqiang Zhu、Chenjia Bai、Xuelong Li
- **机构：** 中国电信人工智能研究院（TeleAI）、伽马智能（Gamma Robotics）
- **项目：** https://teamillusion-smart.github.io/
- **数据：** https://huggingface.co/datasets/TeleEmbodied/SMART-Data
- **权重：** https://huggingface.co/TeleEmbodied/SMART-VLA
- **共享模型实现：** https://github.com/TeleHuman/PRTS（不等于 SMART-Sim 数据生成器）
- **入库日期：** 2026-11-11
- **核对范围：** 通读 arXiv HTML v1 正文（问题、SMART-Sim、任务合成、分布式渲染、SMART-VLA 训练、消融、仿真与三平台真机评估、结论）及相关实验/方法附录；交叉核对 TeleEmbodied 官方数据集、权重卡和 PRTS README/关键运行入口。未逐篇核查参考文献；官方网站本轮访问受限，以论文及官方 HF 页面核验开放状态。
- **通读：** 已完成

## 研究问题与方法

仅靠真实示范收集铰接物体的开合、滑动、操纵难以覆盖不同铰链/滑轨的几何、可达性和视觉变化。SMART 构建以 Isaac Lab 为基础的 SMART-Sim：对分节物体与关节标注功能结构，生成满足关节约束与接触条件的轨迹，再通过视觉与物理域随机化、agent 生成任务程序和分布式渲染制成视觉—动作示范。

1. **资产与动作规划：** 铰接关节/可操作部件标注；运动生成结合 cuRobo 无碰撞路径与针对接触/关节约束的求解、仿真成功校验。
2. **任务合成：** agent 基于可用资产及 23 种可复用操作技能组合原子与复合任务，并生成条件/校验规则；任务自动生成仍可能出现幻觉。
3. **分布式合成：** 先采集成功运动序列，再重放采样视觉与物理变化、渲染多视角数据并转换 LeRobot 数据集。并行设计采用调度服务和多类工作进程。
4. **训练：** SMART-VLA 基于 Qwen3-VL 与动作专家。阶段一在大规模模拟示范上使用动作 token 预测预训练；阶段二在下游合成任务上使用 DiT / flow-matching 动作生成。论文明确没有使用 PRTS 的 CRL 对比目标，不应混为一谈。
5. **部署：** 在三类真实机器人上测试纯仿真训练形成的策略，不依赖任务真机示范；仍需模拟任务设置及资产工程，不能外推为任意机器人/物体免适配。

## 数据与评测口径

| 类别 | 原文报告 |
|---|---|
| 数据规模 | 超过 100 万条轨迹、约 5 亿图像帧、超过 4,600 小时 |
| 资产/场景 | 2,507 件铰接资产、1,122 个场景；资产库 88 个功能类别，发布集覆盖其中 23 类 |
| 技能与任务 | 23 个操作技能、44 种原子任务类型；5 种采样机器人配置 |
| LIBERO | SMART-VLA 平均成功率 96.8%；无大规模预训练 94.7% |
| RoboCasa365 | 28.8%；无阶段一预训练 20.0%；Composite Unseen 仅 7.5% |
| 真机 | AC-1 / R1Pro / 双臂 RealMan RM75，12 项任务，每项 15 次，平均成功率 76.7%；无合成预训练 50.0% |

真机平均差异为 **+26.7 个百分点**；属特定平台/任务/试验次数的论文结果，不代表通用机器人操作水平。

## 边界与局限

- **仿真平台源码未见公开完整实现。** 官方提供 SMART-Data（LeRobot v3.0）、SMART-VLA 权重；PRTS 仓库提供通用模型后训练与推理入口，不能替代完整 SMART-Sim 资产标注、任务生成和分布式采样流程。
- **数据格式注意：** SMART-Data 是 LeRobot v3.0，PRTS 当前 README 示例使用 v2.1，不能不验证兼容性就直接按原命令运行。
- **论文局限：** 长时序轨迹合成随规划失败率提高而降速；任务生成智能体可能产生幻觉；资产关节/功能标注仍涉及大量人工后处理。
- **评测局限：** RoboCasa 未见任务族效果低，真机仅作者平台及指定任务；不能声称普适零样本迁移。

## 对 wiki 的映射

- [SMART / Gamma-Art 实体详情](../../wiki/entities/paper-gamma-art-smart.md)
- [Sim2Real 原理](../../wiki/concepts/sim2real.md)
- [机器人操作任务](../../wiki/tasks/manipulation.md)

## 一手资料

- [论文全文 v1](https://arxiv.org/html/2610.07652v1)
- [官方数据集](https://huggingface.co/datasets/TeleEmbodied/SMART-Data)
- [官方模型权重](https://huggingface.co/TeleEmbodied/SMART-VLA)
- [共享 PRTS 实现仓库](https://github.com/TeleHuman/PRTS)
