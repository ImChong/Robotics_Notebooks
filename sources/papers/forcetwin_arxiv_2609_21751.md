# ForceTwin：仪器化人类交互的物理信息数字孪生（arXiv:2609.21751）

> 来源归档（ingest）

- **英文标题：** ForceTwin: Physics-informed Digital Twins for Robotic Manipulation from Instrumented Human Interaction
- **标题：** ForceTwin：仪器化人类交互的物理信息数字孪生
- **类型：** paper
- **作者：** Tim Engelbracht, René Zurbrügg, Mayank Mittal, Marco Hutter, Marc Pollefeys, Hermann Blum, Zuria Bauer（† Tim Engelbracht 通讯）
- **机构：** ETH Zurich；NVIDIA；Microsoft；University of Bonn
- **arXiv：** <https://arxiv.org/abs/2609.21751>
- **PDF：** <https://arxiv.org/pdf/2609.21751>
- **项目页：** <https://timengelbracht.github.io/forcetwin-website/>
- **开源：** 项目页 **未见** GitHub / 数据 / 权重（复核 **2026-09-28**）；BibTeX 标注 preprint 上线后提供
- **入库日期：** 2026-09-28
- **配套站点：** [forcetwin-website.md](../sites/forcetwin-website.md)

## 核心论文摘录

### 1) 手持力传感夹爪 → 同步位姿与接触 wrench

- 人在场即可探激铰接物体，无需先部署机器人；补偿夹爪自重/惯性后得到物体侧接触 wrench。
- **对 wiki 的映射：** [../../wiki/entities/paper-forcetwin.md](../../wiki/entities/paper-forcetwin.md)

### 2) 半参数动力学：$I$、Coulomb/粘性摩擦 + 结构化神经残差

- BIC 选 revolute/prismatic；因子图精化关节几何与 $q_i$；机制力（闭门器、弹簧挡）用状态相关残差，而非常数参数。
- **对 wiki 的映射：** [../../wiki/entities/paper-forcetwin.md](../../wiki/entities/paper-forcetwin.md)

### 3) 阻抗控制前馈 + 仿真资产导出

- Spot / Franka FR3 上 9 组 object–embodiment：**87%** 目标完成 vs VLM 先验 **60%**、纯运动学孪生 **57%**；强机制物体上基线易卡死。
- 同一孪生用于全身穿门策略训练并真机部署；惯性参数误差相对 VLM 先验约减半。
- **对 wiki 的映射：** [../../wiki/entities/paper-forcetwin.md](../../wiki/entities/paper-forcetwin.md)

## 当前提炼状态

- [x] sources + 项目页核查
- [x] wiki 实体页
