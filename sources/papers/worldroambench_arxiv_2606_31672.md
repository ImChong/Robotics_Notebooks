# WorldRoamBench: An Open-World Benchmark for Long-Horizon Stability of Interactive World Models

> 来源归档（ingest）

- **标题：** WorldRoamBench: An Open-World Benchmark for Long-Horizon Stability of Interactive World Models
- **类型：** paper / benchmark / interactive world model
- **arXiv：** <https://arxiv.org/abs/2606.31672>
- **PDF：** <https://arxiv.org/pdf/2606.31672>
- **项目页 / 榜单：** <https://worldroam.amap.com/> · [Leaderboard](https://worldroam.amap.com/#leaderboard)
- **机构：** 高德地图视觉技术实验室（Amap CV Lab）、南京大学（NJU）、清华大学（THU）、北京大学（PKU）
- **入库日期：** 2026-09-15
- **步骤 2.5（开源核查）：** **部分开源** — 在线 Leaderboard + 模型提交入口 **已开放**；评测数据集 OSS 直链可下载（`worldroam.zip`，约 2.5GB）；**GitHub 评测代码** 项目页标为 coming soon（截至入库日未挂公开仓库）。

## 摘要级要点

- **问题：** 交互世界模型（IWM）进展快，但既有基准多在 **轨迹级** 评 action following，忽视 **记忆** 与 **交互物理**；长程稳定性缺统一坐标系。
- **规模：** **1000+** 测试用例（论文摘要写 600+ test cases；站点写 1000+）；**Nature / Urban / Indoor**；**第一/第三人称**；游戏世界 + 真实世界；**WASD** 连续交互 **10–60s**。
- **四维评测（各含专门创新）：**
  1. **Action Following** — **逐帧** action metric（pose estimation + latent-stride 离散化），暴露轨迹对齐掩盖的逐步失败；含 Act Acc / Part Acc / TrajScore / nATE。
  2. **Visual Quality** — 美学（LAION aesthetic）+ 成像（MUSIQ）+ **分段 drift**（滑窗内相对峰值下降，捕捉中段崩塌）。
  3. **Interaction Physics** — **可控性门控** 下评力学（碰撞/穿模/地形/重力/形变）、光学（反射/阴影）、3D 一致性。
  4. **Memory** — **与 action 解耦**：场景记忆用 **过渡局部 3D 点云重建**（Retention / Anti-Hallucination）；主体记忆用 SAM2 跟踪 + VLM。
- **关键发现（项目页）：** TrajScore 高 ≠ 逐帧动作对；视觉好 ≠ 动作跟得好；物理更严可能牺牲动作跟随；帧对记忆评测会被 action 误差污染。
- **榜单快照（截至 2026-08-20，Overall）：** Genie 3 **70.32** > Lyra 2.0 **69.61** > HappyOyster **68.53**；开源榜 Lyra 2.0 领先；**无一模型** 四维同时可靠。

## 对 wiki 的映射

- 新建实体：[paper-worldroambench](../../wiki/entities/paper-worldroambench.md)
- 项目页归档：[worldroambench.md](../sites/worldroambench.md)
- 交叉：[paper-abot-world-0](../../wiki/entities/paper-abot-world-0.md)、[paper-harnesseval-w](../../wiki/entities/paper-harnesseval-w.md)、[hub-embodied-eval-benchmark](../../wiki/overview/hub-embodied-eval-benchmark.md)、[generative-world-models](../../wiki/methods/generative-world-models.md)

## 参考来源（原始）

- arXiv:2606.31672
- 项目页与 Leaderboard：<https://worldroam.amap.com/>
