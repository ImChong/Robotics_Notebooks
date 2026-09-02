# Non-Prehensile Throwing: A Reinforcement Learning Perspective（arXiv:2609.00771）

> 来源归档（ingest）

- **标题：** Non-Prehensile Throwing: A Reinforcement Learning Perspective
- **短名：** NP-Throw
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.00771>
- **PDF：** <https://arxiv.org/pdf/2609.00771>
- **项目页：** <https://abdullah-aist.github.io/NP-Throw/>
- **代码：** <https://github.com/Abdullah-AIST/NP-Throw>
- **机构：** 日本产业技术综合研究所（AIST）；早稻田大学（Waseda University）
- **入库日期：** 2026-09-02
- **索引来源：** [具身智能小站 7 篇盘点](../blogs/wechat_embodied_station_7_papers_contact_manipulation_2026-09-02.md)
- **一句话说明：** RL 直接优化关节轨迹，利用滑动/滚动接触；仿真 99% 成功率，UR5e 零样本真机 97%。

## 开源状态（步骤 2.5，2026-09-02）

- **已开源**：`Abdullah-AIST/NP-Throw`。

## 核心摘录（面向 wiki 编译）

### 摘录 1：方法与接触模式

- 无需解析接触模型或低维轨迹参数化；MDP 以目标、物体模型与初始构型为条件演化关节状态；离线规划 joint-jerk 轨迹并上采样部署。
- **对 wiki 的映射：** [paper-np-throw](../../wiki/entities/paper-np-throw.md)

### 摘录 2：sim-to-real

- minimum-jerk 系统辨识 + 不确定性感知训练；UR5e 近物理极限（5 m/s 末端速度）投掷 790 g 重物与 20×20×28 cm 大物体至 350 cm 距离或 180 cm 高度。
- **对 wiki 的映射：** [paper-np-throw](../../wiki/entities/paper-np-throw.md)

## 当前提炼状态

- [x] 项目页/仓库已交叉核查
- [x] wiki 映射：`wiki/entities/paper-np-throw.md` 新建
