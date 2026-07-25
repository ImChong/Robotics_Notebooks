# A Low Cost Modular Actuator for Dynamic Robots

> 来源归档

- **标题：** A Low Cost Modular Actuator for Dynamic Robots
- **类型：** paper（MIT 硕士论文 / thesis）
- **作者：** Benjamin G. Katz
- **导师：** Sangbae Kim
- **机构：** 麻省理工学院机械工程系（MIT Department of Mechanical Engineering）
- **年份：** 2018（S.M.，提交日期 2018-05-11）
- **DSpace：** https://dspace.mit.edu/entities/publication/b85069e2-f1cd-470e-a92a-9bf0dadfa7ee
- **Persistent handle：** http://hdl.handle.net/1721.1/118671
- **PDF：** https://dspace.mit.edu/bitstream/handle/1721.1/118671/1057343368-MIT.pdf（bitstream UUID `f3404f79-1cb4-4e61-a51b-1f4ef8bb4e45`；MD5 `46b87d00bb5d7d665c0e2b676055d995`；约 28.8 MB）
- **OCLC：** 1057343368
- **作者博客叙事：** https://robot-daycare.com/posts/2019-03-04-hello-there-mini-cheetah/
- **代码（附录 A，部分开源）：**
  - 电机驱动硬件：https://github.com/bgkatz/3phase_integrated（MIT）
  - 电机驱动固件（mbed）：https://os.mbed.com/users/benkatz/code/Hobbyking_Cheetah_Compact/（DRV8323 新版：`HKC_MiniCheetah`）
  - 非 mbed 固件：https://github.com/bgkatz/motorcontrol（MIT）
  - SPIne 硬件：https://github.com/bgkatz/SPIne
  - SPIne 固件：https://os.mbed.com/users/benkatz/code/SPIne/
  - 执行器表征数据与分析：https://github.com/bgkatz/actuator
- **版权注意：** MIT 学位论文受版权保护；可从 DSpace 查看/下载/打印，**未经书面许可不得再分发**。本仓库只存链接与 MD5，不入库 PDF 二进制。
- **入库日期：** 2026-07-25
- **一句话说明：** MIT Mini Cheetah 系模块化 QDD 执行器的奠基工程文档：成品航模外转子电机 + **6:1** 单级行星 + 集成 FOC/磁编/CAN，BOM≈$300；装成 9 kg / 12 DoF 四足并完成离线轨迹优化后空翻。
- **开源状态：** **部分开源** — 附录 A 公开驱动 PCB、固件、SPIne 与表征数据；**未**列出执行器壳体/行星箱机械 CAD 或整机 CAD。后续社区有 `mit-biomimetics/Cheetah-Software` 等控制栈，但不在本 thesis 附录内。
- **沉淀到 wiki：** [paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)

---

## 核心贡献（摘录）

1. **模块化本体感受执行器（480 g）**：U8 级 COTS 外转子 BLDC（约 $60–90）+ **6:1** 现成行星 + 集成三相逆变器（24 V 标称 / 40 A 峰值相电流设计）+ AS5047P 磁编 + 双 XT-30 / 双 CAN 菊花链；峰值 **17 N·m**、连续 **6.9 N·m**、输出惯量 **0.0023 kg·m²**、最大输出转速约 **40 rad/s @ 24 V**。
2. **成本与集成**：子 50 件量级 BOM 约 **$300**；整机硬件成本低于 Cheetah 系列单执行器。外壳可直接承受较大弯矩，便于肢体直连。
3. **冲击/传动分析**：用输入/输出/末端柔顺三案例估计碰撞载荷；行星太阳轮许用约 11 N·m；指出传动需按冲击而非仅按电机峰值力矩选型。
4. **电流环与标定**：离散时间 PI 电流控制；10 A / 20 A 相电流上升时间约 **75 µs / 110 µs**；磁编非线性与齿槽查表标定；弱磁可再提约 **20%** 转速与 **7%** 峰值功率。
5. **台架表征**：四象限测功机；传动效率约 **90–95%**；静摩擦约 0.09 N·m、力矩相关摩擦约 0.04 N·m/N·m；绕组热阻约 1.23 K/W，平均耗散宜 &lt; **60 W**（绕组 &lt;100 °C）；小风扇可使热阻降至约 **0.34 K/W**，连续力矩可近翻倍。
6. **四足平台（≈ Mini Cheetah 前身）**：约 Cheetah 3 的 60% 尺度，含电池约 **9 kg**，单腿竖直力约 **1.6** 倍体重；UP Board + SPIne（四路 CAN）架构。
7. **后空翻**：2D 矢状面非线性轨迹优化（CasADi / IPOPT 类工具链）离线生成力矩，真机开环回放 + 关节 PD；COM 峰值高度约 **0.65 m**，机械输出峰值约 **690 W**；着陆用宽站姿高阻尼 PD。作者称据其知识为四足首次完整空翻。
8. **其它落地**：双边遥操作/力反馈臂；6 DoF 下半身双足（Little Hermes / 平衡反馈）。

## 对 wiki 的映射

- 论文实体页：[paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)
- 范式对照（Cheetah 本体感受）：[paper-notebook-proprioceptive-actuator-design-in-the-mit-cheeta](../../wiki/entities/paper-notebook-proprioceptive-actuator-design-in-the-mit-cheeta.md)
- 开源 QDD 对比：[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- 力矩电机纵深：[depth-torque-motor-design](../../roadmap/depth-torque-motor-design.md)
- 驱动/硬件仓：[bgkatz_3phase_integrated](../repos/bgkatz_3phase_integrated.md)、[bgkatz_motorcontrol](../repos/bgkatz_motorcontrol.md)、[bgkatz_spine](../repos/bgkatz_spine.md)
- 作者叙事：[robot_daycare_mini_cheetah](../sites/robot_daycare_mini_cheetah.md)
