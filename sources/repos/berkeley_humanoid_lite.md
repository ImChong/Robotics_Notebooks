# Berkeley-Humanoid-Lite

> 来源归档

- **标题：** Berkeley Humanoid Lite
- **类型：** repo
- **组织：** HybridRobotics
- **链接：** https://github.com/HybridRobotics/Berkeley-Humanoid-Lite
- **项目页：** https://lite.berkeley-humanoid.org/
- **文档：** https://berkeley-humanoid-lite.gitbook.io/docs
- **论文：** https://arxiv.org/abs/2504.17249
- **许可：** MIT
- **星标（截至 2026-07-25）：** ~1442
- **入库日期：** 2026-07-25
- **一句话说明：** 低成本 3D 打印开源人形：整机 CAD、摆线关节、电机/控制参数、BOM、底层控制、Isaac Lab 训练与实机部署。
- **开源状态：** **已开源**（主仓 + 门户 + 文档；执行器为成品电机 + 3D 打印摆线）
- **项目页归档：** [sources/sites/berkeley_humanoid_lite.md](../sites/berkeley_humanoid_lite.md)
- **沉淀到 wiki：** [berkeley-humanoid-lite](../../wiki/entities/berkeley-humanoid-lite.md)

---

## 关节相关要点（策展）

| 项 | 内容 |
|----|------|
| 减速比 | 约 **15:1** 摆线 |
| 电机 | 外转子；约 **14 对极**；力矩常数约 **0.1176 N·m/A** |
| 控制 | 公开电流环 / 位置环 / 速度环等参数 |
| 链路 | 设计 → RL（Isaac Lab）→ 实机 |

## 局限（官方叙事 + 策展）

- 当前 **3D 打印摆线** 在高性能运动中偏脆弱；后续版本可能改成品关节。
- 适合学习与验证，**不适合原样用于重型人形**。

## 对 wiki 的映射

- [Berkeley Humanoid Lite](../../wiki/entities/berkeley-humanoid-lite.md)
- [开源 QDD 执行器项目对比](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
