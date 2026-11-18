# Why Walking Robots Are So Hard to Build（行走机器人执行器自研为什么困难）

> 视频来源归档。本页为 Video Summarizer for YouTube 生成的章节摘要之整理，**不是逐字稿，亦非已完整观看的原片核对**。

- **标题：** Why Walking Robots Are So Hard to Build
- **类型：** video
- **视频：** https://www.youtube.com/watch?v=qKkivaZwqTo
- **关联代码：** https://github.com/the5439workshop/Actuator_2
- **时长：** 18:39（1119 秒）
- **入库日期：** 2026-11-18
- **核对范围：** 视频摘要接口返回的 0:00–17:40 章节结构与要点；逐字稿接口返回本周额度不足，未取得可核对的完整字幕或原片台架数据。
- **通读：** 未完成
- **沉淀到 wiki：** [Actuator 2.0](../../wiki/entities/the5439workshop-actuator-2.md)

## 章节脉络（来自视频摘要，待逐字复核）

| 时间 | 内容 | 技术判断 |
|------|------|----------|
| [0:47](https://youtu.be/qKkivaZwqTo?t=47) | 1.0 的 3D 打印拉索/卷筒结构负载时出现滑移、断裂，摘要称峰值约 10 N·m | 高扭矩关节需同时考虑传动和结构刚度 |
| [2:21](https://youtu.be/qKkivaZwqTo?t=141) | 2.0 改铝制结构，将行星减速机构融入定子中空空间 | 集成体积与可制造性共同影响架构 |
| [5:06](https://youtu.be/qKkivaZwqTo?t=306) | 7075 铝齿轮的 CNC 加工 | 精度、公差、材料强度与寿命分开评估 |
| [6:40](https://youtu.be/qKkivaZwqTo?t=400) | 行星架改为分件装配，摘要称约 6:1 减速 | 先保证多级零件对位，不把设计比视为最终实测 |
| [11:45](https://youtu.be/qKkivaZwqTo?t=705) | 定子绕线、磁钢转子、编码器、温度传感器及驱动器集成；摘要称成品约 900 g | 装配和热设计是执行器整体问题 |
| [13:51](https://youtu.be/qKkivaZwqTo?t=831) | 初始力矩测试不达预期 | 必须重新检查电气假设与标定 |
| [14:28](https://youtu.be/qKkivaZwqTo?t=868) | 讨论铜耗与电流的平方关系，以及绕组匝数的权衡 | 固定铜量条件下需比较铜损和扭矩，不能只数匝数 |
| [17:40](https://youtu.be/qKkivaZwqTo?t=1060) | 摘要记载最终约 20 N·m、100 W 的测试结果 | **100 W 的测量口径、持续时长、输出轴工况待核实** |

## 工程结论与证据界限

- 核心价值是一次自研关节从拉索传动失败，转向中空定子内集成行星传动、加工和调试的迭代记录。
- **尚不能将 20 N·m / 100 W 解释为连续额定扭矩或效率指标。** 视频摘要未给定输出转速、扭矩计校准和热稳态条件。
- 可独立核实的软件/物料事实见 [官方仓库归档](../repos/the5439workshop_actuator_2.md)；视频主张待取得字幕或重新核对原片后再补齐证据。
