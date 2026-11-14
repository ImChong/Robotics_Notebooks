# D-JEPA（arXiv:2609.24749）

> 来源归档（paper）

- **标题：** D-JEPA: A Decision-Aligned Latent World Model
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24749>
- **PDF：** <https://arxiv.org/pdf/2609.24749>
- **项目页：** <https://nebulis-lab.com/D-JEPA/>
- **代码：** <https://github.com/NEBULIS-Lab/D-JEPA>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 用实际执行结果学习候选未来之间的关系，修正潜在距离与任务成败不一致的问题，让世界模型选对能成功的动作。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库可达（git ls-remote），项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 预测接近目标的潜在状态不等于选对动作；潜在距离与成败可能不一致。
- 以真实执行结果监督候选未来之间的排序关系，使潜在空间与决策对齐。
- 指标：公众号提示：决策对齐在匹配候选集合下评估，效果须连同动作候选生成方式理解。

## 对 wiki 的映射

- [paper-d-jepa-decision-aligned-latent-world-model](../../wiki/entities/paper-d-jepa-decision-aligned-latent-world-model.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)

## 全文通读复核（2026-11-13）

- **核对范围：** arXiv:2609.24749v1（2026-09-21）正文§1–§7全文及必要附录A/B（协议、目标、任务接口），图表文字层、参考文献浏览；另读项目页、根README及固定版本模块与cached replay源码。未运行仿真/模型。
- **通读：** 已完成
- 训练以同初态候选的执行结果监督，部署只用预测特征/序数、无outcome标签；这是有监督决策对齐，不是无标签JEPA预测自动变好。拟合/校准/测试按start身份隔离。
- ±.2修正保证跨base gap>.4的顺序不翻转，仅是score信赖范围，不是任务安全保证。Exact ordinal realization将排序写入terminal半径，100%恢复rank是代数接口一致性，不是100%任务成功、重新学习物理或单骨干蒸馏。
- PushT225/256=87.89为校准组合，ordinal realization223/256=87.11与relational一致，不能把最高组合分放到ordinal模式。confirmation按预先规则筛出非平凡决策边界，不是随机全部PushT状态。
- Granular CD .358不优于DINO .341，严格阈值成功率更高不代表全部指标优。Driving七个困难场景、三source logs、平均PDMS95.34不是全NAVSIM成功率；真机两任务100配对trial为81对64。
- 形状是未见几何，七视觉条件家族已参与task-local fitting、仅start新；不能称视觉扰动种类零样本。Reacher transport只作校准表示诊断，物理成功来自relational选择。
- 当前轻量包能CPU cached-feature replay；实际sim/robot执行是另一流程，robot设备controller/calibration需使用者提供，未在本次运行。
