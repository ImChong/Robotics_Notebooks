# Zero-Shot Compositional Generalization for Embodied Manipulation（具身操作的零样本组合泛化）

> 论文一手资料归档；所有实验数字均为作者报告的仿真评测，而非本库独立复现。

- **类型：** paper
- **英文标题：** Zero-Shot Compositional Generalization for Embodied Manipulation
- **中文标题：** 具身操作的零样本组合泛化
- **论文：** https://arxiv.org/abs/2611.22797
- **HTML 原文：** https://arxiv.org/html/2611.22797v1
- **PDF：** https://arxiv.org/pdf/2611.22797v1
- **版本：** arXiv:2611.22797v1，2026-11-17，cs.RO 预印本
- **作者：** Baicheng Li, Wenkang Qin, Youqi Pan, Dongxiao Mao, Yukun Zhou, Chuanfu Shen, Dong Wu, Shunkai Zhou, Zike Yan, Wei Sui, Hongbin Zha
- **机构：** 北京大学（Peking University）；地瓜机器人（D-Robotics）；电子科技大学深圳高等研究院（Shenzhen Institute for Advanced Study, UESTC）；香港中文大学香港具身智能实验室（Hong Kong Embodied AI Lab, CUHK）
- **项目/源码：** 本版 arXiv 页面及全文没有列出作者官方项目页或公开训练/推理仓库；截至 2026-11-20 检索未核实独立官方实现，不能标为已开源。
- **入库日期：** 2026-11-20
- **核对范围：** arXiv v1 官方 HTML 全文，第 1–6 节、两张主表、图 1–6 图注及 References；核对问题设定、标定 FK、机器人接口预训练、正向可达 IK witness、策略阶段损失、跨 backbone/跨模拟器结果、结论。论文提及的 supplementary implementation details 未作为独立可检索附件核对；未运行模型或仿真复现。
- **通读：** 已完成
- **一句话：** 把示范关节动作映射为已标定的工具空间目标，以可达性筛选后监督未见该任务的其他机器人原生解码器，并在视觉策略学习时额外加入 FK 约束，实现已知机器人与已知技能的未见组合迁移。
- **Wiki 映射：** [独立论文实体](../../wiki/entities/paper-zero-shot-compositional-generalization-manipulation.md)

## 1. 研究问题与零样本定义

示范形成机器人集合与技能集合的二部图，只观察部分组合；评测组合不得出现在动作接口预训练和视觉策略训练的示范中。**每个待测机器人和技能本身在训练期都出现过**，所以这是已知机器人×已知技能的缺失边补全，不是全新机器人或全新任务泛化。Franka Panda、UR5、Piper 为文中举例本体；论文用 RoboTwin 和 ManiSkill3（Push）做系统评测。

跨机器人时本体原生关节维度、关节定义及安装差异很大。把关节向量 padding 到同一长度只能解决张量尺寸，不能建立真实物理对应。

## 2. 从源关节动作到异构本体的几何监督

1. 对源机械臂示范动作计算工具中心点在共享任务坐标系的目标，包括位置、姿态、归一化夹爪开度。标定关系为安装变换 × 机械臂 FK × 法兰到工具标定。
2. 源动作经本体输入映射与共享变分编码器生成**每时刻 32 维**潜变量，再通过源/外部机器人各自的 native decoder 产生原生关节命令；动作块是每时刻潜变量的序列。
3. 训练源分支时结合标准化关节重建、源工具位姿重建及 KL；训练外部目标本体时，通过可微 FK 比较其解码工具目标与源示范工具目标，加位置、旋转、开度误差。
4. **仅正向 IK 可达性 witness 成立的目标施加异构监督**：数值 IK 找到满足关节限位与位姿容差的有限解才标记可行；求解失败只是“不监督”，并非理论上不可达。
5. 源本体原生重建保留其动作构型先验；几何监督解决只靠共享编码器+本体重建留下的**不同本体潜坐标语义不确定性**（论文以各本体 signed permutation 不变性说明）。

## 3. 视觉语言策略训练及闭环

动作接口训练结束后冻结共享编码器与所有本体原生解码器。视觉策略输入图像、语言、当前本体状态，以训练集拟合的归一化潜动作目标进行 flow-matching 训练；推理时预测动作块，反归一化后由当前机器人专属 decoder 生成**绝对关节位置和夹爪开度**，执行前缀再重新观测。

Full 在策略阶段通过可微的 clean action 估计、冻结 decoder 和 FK 加入几何损失：同时比较源和可行外部机器人的工具效果与**同一源示范**工具目标；不是给未见组合引入目标机器人专家关节标签。

- Source：仅本体重建的接口，策略无附加 FK loss。
- Cross：接口启用 foreign FK 监督，策略无附加 FK loss。
- Full：Cross 接口 + 策略阶段 FK loss。
- Direct EEF + IK：直接预测工具空间命令，在线数值 IK 转为原生命令，视觉和语言条件对齐；不是学习 native decoder。

## 4. 实验数据：分开模型、模拟器与消融口径

表 1：各任务 1000 次评测 episode，seen/held 均为任务组平均成功率。以下只列 **Held**（%）：

| Simulator / backbone | Native joints | Full | 增益（百分点） |
|---|---:|---:|---:|
| RoboTwin / π0.5 | 17.0 | 33.6 | +16.6 |
| RoboTwin / OpenWAM | 34.8 | 46.5 | +11.7 |
| RoboTwin / FastWAM | 9.7 | 29.8 | +20.1 |
| ManiSkill3 Push / π0.5 | 3.1 | 35.9 | +32.8 |
| ManiSkill3 Push / OpenWAM | 11.6 | 43.5 | +31.9 |
| ManiSkill3 Push / FastWAM | 5.4 | 28.6 | +23.2 |

表 2，另一个受控对照的 **Held**（%），不可直接混为表 1 的相同训练结果：

| 方案 | OpenWAM | FastWAM |
|---|---:|---:|
| Direct EEF + IK | 41.5 | 25.2 |
| Source | 33.4 | 17.7 |
| Cross | 42.9 | 26.3 |
| Full | 46.5 | 29.8 |

- Foreign FK 监督在该受控组让 Cross 比 Source 提升 **+9.5 / +8.6 pp**；策略级几何损失让 Full 比 Cross 再提升 **+3.6 / +3.5 pp**。
- 在**独立**三个配对策略运行中，Cross 对 Source 的 Held 提升为 **19/16/11 pp**，平均 15.3 pp；该面板不能当作表 2 的误差条。
- 可达目标的外本体 TCP 误差 Source/Cross 为 **302.13 / 4.37 mm**；它是动作实现的诊断，不等于任务最终完成率。
- 跨本体 command-generation 的配对基准上，Cross 的模型侧中位时延相对 Direct EEF + IK 快 **1.71×**；包含图像编码、flow 预测与 native command 生成，不可等同控制伺服频率。

## 5. 工程假设与局限

- 要知道每台机器人的关节限位、安装坐标、真实 FK 与工具外参；错误标定会破坏跨本体共享目标。
- 可达 IK witness 是有正解的充分见证，失败不证明无解；并不为路径碰撞、冗余配置连续性、接触成功、动力学可行性提供保证。
- 研究核心是**未观测 robot–skill 配对**，不是未知新机器人或未经见过的技能；实验来自两套模拟器，不能视为已证明真机部署安全性。
- 论文第 4 节提及 supplementary implementation details，但公开的 v1 HTML 主文未展示完整参数/求解器设置；在缺少可核实官方实现的情况下，不绘制虚构源码调用时序图。

## 6. 知识库映射

- [独立论文实体](../../wiki/entities/paper-zero-shot-compositional-generalization-manipulation.md)
- [跨本体迁移综述](../../wiki/overview/hub-cross-embodiment.md)
- [跨本体纵深路线 Stage 1](../../roadmap/depth-cross-embodiment.md)

## 一手资料

- [论文 arXiv v1](https://arxiv.org/abs/2611.22797v1)
- [arXiv 官方 HTML 全文](https://arxiv.org/html/2611.22797v1)
