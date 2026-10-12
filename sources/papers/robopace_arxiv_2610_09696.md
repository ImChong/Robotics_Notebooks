# RoboPace 论文来源归档

## 元数据

- **题目：** RoboPace: Contact-Aware Time-Optimal Retiming for Action-Chunk Policies
- **方法：** RoboPace（在线接触感知重定时层）
- **作者：** Mimo Shirasaka, Takehiko Ohkawa, Takuya Okubo, Nicola Scianca, Tatsuya Matsushima, Kei Ota
- **单位：** arXiv 页与 HTML 均未列出（域名 airoa.io 提示与 AIRoA 相关，未核实）
- **预印本：** https://arxiv.org/abs/2610.09696
- **HTML：** https://arxiv.org/html/2610.09696v1
- **版本：** v1，2026-10-07，cs.RO
- **项目页：** https://robopace.airoa.io/ （2026-10-12 抓取时域名解析失败，内容未核实；见 [robopace-project.md](../sites/robopace-project.md)）
- **代码：** arXiv 摘要页无 GitHub / HF 链接，论文未声明代码或权重发布。

## 论文要解决的问题

无机器人演示（UMI 式手持采集、直接录人手）训练的 VLA 继承了人类的动作时序：人手柔顺，能快速接触；机器人接触时可能过冲，而在自由空间里反而可以比人更快。统一加速或减速都不理想。

## 方法与训练信息

1. 在线重定时层：保持策略输出的几何路径不变，只调整时间参数化；无需重新训练，实时运行。
2. 重定时用 TOPP-RA（时间最优路径参数化）逐 chunk 求解，约束为关节速度/加速度上限，加上接触关键路点上的笛卡尔手速上限 v_contact（转为沿路径的关节速度上界，ρ = 4 源步的余弦斜坡过渡）。
3. 接触线索：几何线索（双手距离 < 0.08 m 或距桌面 < 0.02 m）；学习线索：EfficientNet-B0 + GRU（4.69M 参数）从腕部相机与夹爪角度预测距接触的源步数（stc），stc ≤ 20 步（扣延迟后）则标记；训练用 8,601 条录制，按录制划分。接触起始预测误差约 0.80 源步（0.027 s）。
4. 运行时：阻尼最小二乘 IK 转关节路径；每执行 16 个源步重新查询策略；新 chunk 以 linear / cosine / tanh 调度并入在飞计划。每 chunk 计算耗时论文未报告。
5. 策略：基于 π0.5 端到端微调，动作视野 32 步，两个腕部相机，用 8,434 小时 YUBI 录制预训练；平台为双臂 7-DoF OpenArm v1.0 + YUBI 手，RTX 5070 Ti 工作站。

## 论文报告的结果

- 三个接触密集任务（Cup、Socks、Pyramid 三阶段）共 5 个指令，每指令 10 次，共 50 次。
- 成功次数：RoboPace（Linear-N3）32/50；Decelerated（10 Hz 均匀）22/50；Accelerated（50 Hz 均匀）4/50；RACE 式（仅物理极限重定时）1/50。Demo 速度（30 Hz）未给总成功数，只说 RoboPace 在四个指令上更快；Socks 比示范速度慢。
- 在 Decelerated 成功试验中，RoboPace 用时仅其 45–59%（四个指令）。
- v_contact 敏感性（Pyramid 第 2 阶段）：0.01 / 0.02 / 0.04 m/s，成功率随上限升高而下降；最终选 0.02 m/s（Socks 用 0.04）。
- 过渡调度：Linear-N3 32/50，Tanh-N3 27/50，Tanh-N9 25/50。
- 交接任务（Part，单次）：299 源步 48 s，接触门控下速度中位数 0.01 m/s。

## 复现与解释边界

- 仅一种机器人与一类策略（π0.5），5 指令 × 10 次，样本量小；几何线索与相机线索未单独消融。
- 论文引用 UMI 式采集，但并未说明训练数据来自 UMI。
- 相机线索在最终放置前不触发减速；几何线索不建模被抬起容器。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-robopace.md
- 方法入口：wiki/methods/action-chunking.md
- 任务入口：wiki/tasks/bimanual-manipulation.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.09696>
- arXiv HTML：<https://arxiv.org/html/2610.09696v1>
