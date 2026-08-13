# G1 Confined-Space WBP（arXiv:2608.10220）

> 来源归档（ingest）

- **标题：** Whole-Body Planning for Humanoids Navigating Confined Spaces via Self-Collision Avoidance References
- **类型：** paper / whole-body-planning / humanoid / confined-space / unitree-g1 / residual-rl
- **arXiv abs：** <https://arxiv.org/abs/2608.10220>
- **PDF：** <https://arxiv.org/pdf/2608.10220>
- **HTML：** <https://arxiv.org/html/2608.10220>
- **项目页：** <https://carlosiglezb.github.io/confined-space-wbp-humanoid/> — 归档见 [`sources/sites/confined-space-wbp-humanoid-github-io.md`](../sites/confined-space-wbp-humanoid-github-io.md)
- **代码：** 截至入库日项目页未列 GitHub（Paper/arXiv 按钮亦占位）
- **机构：** 德州大学奥斯汀分校（UT Austin）
- **作者：** Carlos Gonzalez、Luis Sentis
- **发表 / 上传：** 2026-08（arXiv:2608.10220；submitted to IEEE）
- **入库日期：** 2026-08-13
- **一句话说明：** 三阶段人形全身规划：可微刚体体积自碰规避引导 → 全阶动力学 TO → 残差 RL 跟踪；在 Unitree G1 上穿越超 NIST 标准的狭窄环境（\(C_r<1.5\)）。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [carlosiglezb.github.io/confined-space-wbp-humanoid](https://carlosiglezb.github.io/confined-space-wbp-humanoid/) | 管线视频、三环境 MuJoCo 演示 |
| 论文 | [arXiv:2608.10220](https://arxiv.org/abs/2608.10220) | 方法与 Table III 成功率 |

## 开源状态（步骤 2.5，截至 2026-08-13）

- **确认未开源 / 待发布：** 项目页 Code 未列出；页首 Paper/arXiv 链接仍为 `aria-disabled` 占位；BibTeX 标注 under review。
- **处理：** wiki「开源状态」写明「截至 2026-08-13 项目页未列代码」；**不适用**源码运行时序图。

## 摘要级要点

- **问题：** 狭窄空间中碰撞自由构型呈窄非凸流形；样条/粒子松弛引导易陷局部最小；模仿数据难获取。
- **方法：** Stage1 环境感知任务空间 TO（Bézier + 可达性）；Stage2 胶囊/球原语可微 SCA；Stage3 全阶动力学 WBP（硬碰撞约束、摩擦锥）；残差 PPO 跟踪计划。
- **结果要点：**
  - 三测试床：Tilted Stairs / Unobstructed Hole / Obstructed Hole；\(C_r\) 约 1.4–2.0
  - Full pipeline：Stairs 与 Unobstructed Hole **10/10**；Obstructed Left/Right **7/10、6/10**
  - 基线（无膝引导样条 / 线性插值）在 Hole 环境 **0/10**
  - 残差策略在 DR + 外推下遍历成功率 **>95%**（仿真）
- **局限：** 接触序列给定；真机验证列为 future work；求解分钟级（约 2–6 min）。

## 对 wiki 的映射

- 新建实体页：[wiki/entities/paper-g1-confined-space-wbp.md](../../wiki/entities/paper-g1-confined-space-wbp.md)
- 交叉：[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[人形 locomotion](../../wiki/tasks/humanoid-locomotion.md)、[Sim2Real](../../wiki/concepts/sim2real.md)
