# asimov-v1

> 来源归档

- **标题：** Asimov v1（开源人形机器人全栈仓库）
- **类型：** repo
- **来源：** Asimov Inc.（GitHub 组织 `asimovinc`）
- **链接：** https://github.com/asimovinc/asimov-v1
- **入库日期：** 2026-05-07
- **一句话说明：** 面向可制造、可仿真、可定制的开源人形机器人 v1：含机械 CAD、电气设计、MuJoCo 模型与板载软件；硬件许可 CERN-OHL-S-2.0，软件 GPL-2.0。
- **沉淀到 wiki：** 是 → [`wiki/entities/asimov-v1.md`](../../wiki/entities/asimov-v1.md)

---

## 为什么值得保留

- **全栈单一入口**：同一仓库聚合机械、电气、仿真与机载软件，便于对照 BOM、装配与仿真资产的一致性。
- **规格清晰**：公开 README 给出身高/质量/关节划分、CAN 与双板计算架构、材料与负载指标，适合做硬件选型与 Sim2Real 边界讨论。
- **开源许可明确**：硬件与软件分许可，便于合规评估与二次分发策略。

## 与本仓库现有资料的关系

- 与 [`wiki/entities/open-source-humanoid-hardware.md`](../../wiki/entities/open-source-humanoid-hardware.md)（开源硬件对比）同属「可复现人形平台」谱系。
- 与 [atom01_hardware.md](atom01_hardware.md)、[roboto_origin.md](roboto_origin.md) 并列：均为「硬件 + 生态」类参考，但 Asimov 更强调商业 DIY Kit 与官方手册/BOM 外链闭环。
- 仿真侧可与 [mujoco.md](mujoco.md) 对照：仓库内提供 MuJoCo 模型，适合作为接触-rich 腿臂协同的前期验证载体。

## 官方延伸资源（外链）

- 装配与物料：[Assembly Manual](https://manual.asimov.inc)、[BOM](https://manual.asimov.inc/v1/bom)
- 商业预约：[DIY Kit](https://asimov.inc/diy-kit)
- 行走训练（mjlab fork）：[asimovinc/asimov-mjlab](https://github.com/asimovinc/asimov-mjlab)（详见 [`asimov-mjlab.md`](asimov-mjlab.md)）
- 全身仿真 MJCF：`sim-model/xmls/asimov.xml`（含 **`left_toe_joint` / `right_toe_joint` 弹簧被动趾**，无对应 `<motor>`）
- 社区支持：README 中列出的 Forum / Discord 等（以官方页面为准）
