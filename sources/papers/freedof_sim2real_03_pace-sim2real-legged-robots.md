# Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots (PACE)

> 来源归档（paper / 自由度FreeDof Sim2Real 44 篇参考文献 [03/44]）

- **标题：** Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots (PACE)
- **类型：** paper
- **出处：** IJRR 2026
- **章节：** 系统辨识（[四条路线梳理](https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg)）
- **arXiv：** <https://arxiv.org/abs/2509.06342>
- **代码：** <https://github.com/leggedrobotics/pace-sim2real>
- **项目页：** <https://pace.filipbjelonic.com/>
- **入库日期：** 2026-09-20
- **开源状态：** 已开源
- **一句话说明：** 悬空 chirp + CMA-ES 辨识紧凑执行器参数，再以 PMSM 物理能量 reward 盲训腿足 RL 并零样本部署。
- **沉淀到 wiki：** [`wiki/entities/paper-pace-sim2real-legged-robots.md`](../../wiki/entities/paper-pace-sim2real-legged-robots.md)

## 核心摘录（归纳）

- 文内执行器 SysID 主线代表；无需力矩传感器即可显著缩小执行器 gap。
- 4n+1 维关节参数 + 全局延迟；固定基座辨识后窄 DR 训练。

## 对 wiki 的映射

- [paper-pace-sim2real-legged-robots](../../wiki/entities/paper-pace-sim2real-legged-robots.md)
- [freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)
- [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md)
