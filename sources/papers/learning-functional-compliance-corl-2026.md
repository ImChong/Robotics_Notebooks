# Functionally Compliant Humanoid Whole-Body Manipulation｜功能性柔顺的人形机器人全身操作

- **类型：** paper
- **标题：** Functionally Compliant Humanoid Whole-Body Manipulation
- **作者：** Gio Huh, Yue Zhuo, Zehua Wang, Alan Yu, Yibo Cheng, Yanbing Han, Gabriel B. Margolis, Ge Yang
- **机构：** FortyFive Labs、麻省理工学院（MIT）、斯坦福大学（Stanford）、加州理工学院（Caltech）
- **会议：** CoRL 2026
- **原文：** <https://openreview.net/forum?id=wALhOwQWnx>
- **PDF：** <https://openreview.net/pdf?id=wALhOwQWnx>
- **作者主页：** <https://zehuaw1.github.io/>
- **入库日期：** 2026-11-25
- **版本：** 用户提供的 CoRL 2026 会议版 PDF，共 20 页；以 PDF 首页正式标题为准，旧收录标题 Learning Functional Compliance in Whole-body Manipulation 保留为检索线索。
- **核对范围：** 正文第 1–9 页、参考文献及附录 A–M（第 12–20 页），包括图 2–9、表 1–3、A2–A7。
- **通读：** 已完成
- **开源状态：** 附录 H–I 给出 profile 接口、算法及三项任务的 Python 示例；未确认完整训练、推理或部署仓库及权重的公开入口。
- **详情：** [功能性柔顺的人形机器人全身操作](../../wiki/entities/paper-learning-functional-compliance.md)

## 原文要点

- §2–3：标量刚度 K = k I；VLM 在执行前根据任务描述和给定几何生成无状态柔顺函数，执行时以估计力和机器人状态调节刚度。
- §3.1–3.2、附录 L：用 IK 生成受力柔顺位姿，PPO 以适配位姿为目标；力与平衡编码器由仿真标签独立监督，控制器梯度不回传至两者。29-DOF G1 策略 50 Hz。
- §4.1：仿真腕部测试的实际刚度中位端点约 50–740 N/m，校准 CHIP 约 90–370 N/m；实际刚度跨度约 2.5 倍。
- §4.3、附录 J–K：每个任务 10 个匹配设置，评分 0/0.5/1；力依赖配置钥匙插入 5.5、书本翻转 7.0、开柜门 6.0，总分 18.5；不是成功率。
- §6：仅验证双腕接触；小力方向估计不可靠；几何与动作参考预先提供，未实现自动在线几何更新或独立多轴刚度控制。
- 附录 H–I、M：同坐标系力与轴、阈值与工作空间校验、失败时禁用 profile；plateau 在低于 1 N 时返回柔软刚度。

## 对 wiki 的映射

- [功能性柔顺的人形机器人全身操作](../../wiki/entities/paper-learning-functional-compliance.md)
- [移动操作](../../wiki/tasks/loco-manipulation.md)
- [接触丰富操作路线](../../roadmap/depth-contact-manipulation.md)：Stage 3 的任务级柔顺控制。
