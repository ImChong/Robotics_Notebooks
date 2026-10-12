# AlphaNIST 论文来源归档

## 元数据

- **题目：** AlphaNIST: Emergent Assembly Skills from Binary Task Rewards
- **作者：** Octi Zhang, Iretiayo Akinola, Rosario Scalise, Mateo Guaman Castro, Patrick Yin, Tobias Widmer, Miles Macklin, Chris Amevor, Dylan Turpin, Dirk Van Gelder, Jan Carius, Ankur Handa, Gavriel State, Yunrong Guo, Yashraj Narang
- **单位：** NVIDIA；University of Washington（Patrick Yin 同时列两者）
- **预印本：** https://arxiv.org/abs/2610.13354
- **HTML：** https://arxiv.org/html/2610.13354v1
- **版本：** v1，2026-10-09，cs.RO
- **项目页：** https://alpha-nist.github.io/ （归档见 sites/alphanist-project.md）
- **代码：** 论文称在评审修订期之后发布代码；项目页与 arXiv 页均无仓库链接，截至 2026-10-12 未发布。
- **读取范围：** arXiv 摘要页与 HTML 全文抽取；附录细节读到的已写入。

## 论文要解决的问题

能否只用稀疏的二元任务奖励（而非示范、稠密进度塑形或脚本化原语），让机器人学会在 NIST Assembly Task Board #1 上装配零件。该板含 20 个装配对（插销、齿轮啮合、连接器插入、螺母旋入等），最小零件直径仅 4 mm。

## 方法与训练信息

1. **仿真与机器人：** Isaac Lab（GPU 并行环境），仿真 7 自由度 Franka Emika Panda 与平行夹爪（一个主动指、一个被动指）。
2. **观测与动作：** 机器人状态（本体 + 上一动作）及可动零件、配合几何和机器人的表面点云；无离散任务 ID。25 Hz 下输出 7 维相对关节位置目标 + 1 维夹爪指令，隐式阻抗控制器跟踪。
3. **奖励：** 零件到完全装配位姿的距离低于 0.001 m 时给一次 +1 并终止回合；过程无进度奖励；另有固定控制代价（动作幅度、关节速度、力矩）。
4. **算法与网络：** 标准同步 PPO，每环境每迭代 32 步、5 个 epoch、4 个 minibatch；actor 与 critic 各为 SimBa 风格残差网络（LayerNorm + Swish），各带点云编码器。
5. **规模：** 1 到 256K 个并行环境（多卡时每卡 4K）；64K 环境、25K 迭代的共享策略用了 410 GPU 小时。
6. **初始状态：** 改自 OmniReset 的缓存重置库，覆盖未抓取、拾取、搬运、近配合、部分装配等状态，位姿范围大（含倒置）。采用 Beta 分布的成功引导采样，重置概率正比于 ε + √(p̂(1−p̂))。
7. **评测成功：** 完整装配且 15 s 内完成、无安全失败。论文未描述仿真训练里的域随机化。

## 论文报告的结果

- **共享策略：** 64K 环境、25K 迭代，20 个任务总体训练成功率 98.2%；图 5 说明所有任务均超 98%。
- **单任务专家：** 平均 81.4%，总计 16.1K GPU 小时（约为共享策略的 39.3 倍算力）；另训的稠密奖励专家平均 78.3%（原文对 81.4%/78.3% 与奖励类型的对应标注不够明确）。
- **困难任务专家：** M4 螺母旋入、4 mm 销插入、杆插入从随机起点最多 0.3%。
- **联合稠密奖励训练：** 25K 迭代总体 17.8%。
- **环境数扩展（稀疏奖励，Beta/均匀采样）：** 1K 环境 38.6%，4K 78.9%，8K 86.9%，256K 97.4%（Beta）vs 84.3%（均匀）。稠密奖励：1K–4K 环境 11.8–12.0%，256K 77.4%（Beta）vs 34.3%（均匀）。
- **多对装配（表 I，活动装配对数 → 成功率）：** 1:99%，2:97%，3:95%，4:93%，5:92%，6:90%，7:89%，8:87%，9:86%，10:84%。
- **涌现行为：** 利用板面重定向防水连接器；释放重抓对齐（D-sub 连接器、8 mm 销）；侧向笼夹放置齿轮；把 M12 螺母从边缘滚到面上；单侧敲击推进 M4 螺母；铲起杆调整抓取；M16 螺母指尖整圈旋入；多对任务中先把 USB-A 连接器挪开、装杆后再回来插入，M12 旋入时小幅转向避免碰邻近 RJ45。

## 复现与解释边界

- 全部为仿真结果；匹配硬件的速度/力矩限制已加入，但对感知误差、延迟、接触模型失配的鲁棒性未证明。真机评估是计划中的工作。
- 98.2% 是训练成功率（共享策略训练分布上），不是真机部署结果。
- 10 对策略是最大的多对结果；扩展到全部 20 个零件被列为未来工作。
- 资源极大：410 GPU 小时（共享策略）与最高 256K 环境。
- 代码尚未发布，Isaac Lab 之外的重置库来自 OmniReset。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-alphanist.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 概念入口：wiki/concepts/contact-rich-manipulation.md
- 方法入口：wiki/methods/reinforcement-learning.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.13354>
- arXiv HTML：<https://arxiv.org/html/2610.13354v1>
- 项目页：<https://alpha-nist.github.io/>
