# humanoid-touch-dream.github.io（HTD 项目页）

> 来源归档（ingest）

- **标题：** Humanoid Touch Dream — Learning Versatile Humanoid Manipulation with Touch Dreaming
- **类型：** site / project-page
- **官方入口：** <https://humanoid-touch-dream.github.io/>
- **入库日期：** 2026-08-26
- **最后复核：** 2026-09-03
- **一句话说明：** HTD 论文配套站点：接触丰富人形 loco-manipulation 策略、解耦全身控制器跟踪误差对照、浏览器内 MuJoCo WBC Demo，以及遥操作/策略代码发布清单。

## 页面公开资源（检索自 2026-08-26）

| 资源 | URL | 开放程度 |
|------|-----|----------|
| 项目首页 | <https://humanoid-touch-dream.github.io/> | 已公开 |
| arXiv | <https://arxiv.org/abs/2604.13015> | 已公开 |
| PDF | <https://humanoid-touch-dream.github.io/pdfs/humanoid_touch_dream.pdf> | 已公开 |
| 论文仓（入口） | <https://github.com/chrisyrniu/humanoid-touch-dream> | **部分开源**：WBC 已发布；遥操作与 HTD 策略仍 on-going |
| 解耦 WBC 训练/部署 | <https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC> | **已开源**（含 example teacher/student checkpoint） |
| 浏览器 MuJoCo Demo | <https://humanoid-touch-dream.github.io/wbc_mujoco/dist/index.html> | 已公开，导出策略可在浏览器驱动 |

## 项目页核查结论（步骤 2.5）

- **WBC / LBC：** 已开源。Isaac Lab 训练、teacher→student 蒸馏、仿真 play、Unitree G1 真机部署与示例权重均在 [IsaacLab-Decoupled-WBC](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC)。
- **全身遥操作与采数：** 截至 2026-08-26 仍标 on-going（Apple Vision Pro / PICO 仿真遥操作未发布）。
- **HTD 策略训练与部署：** 截至 2026-08-26 仍标 on-going。

## 站点公开主张（便于 wiki 溯源）

1. **系统三层：** RL 全身/下肢控制器 → VR 全身采数 → HTD（视觉+本体+力+触觉 + Touch Dreaming）。
2. **WBC 跟踪：** 相对 AMO（RSS 2025）与 FALCON（L4DC 2026）在多数跟踪误差上更低；尤其高度 \(E_h\)、躯干 yaw \(E_y\)、roll \(E_r\)。FALCON 不跟踪 pitch/roll。
3. **策略结果：** 五任务相对更强 ACT 基线平均成功率 +30.0 个百分点（约 90.9% relative）；latent tactile dreaming 相对 raw tactile 约 30% relative gain。
4. **交互 Demo：** 浏览器内跑导出全身策略，七组运动命令；青色带标训练命令范围。

## 对 wiki 的映射

- [`wiki/entities/paper-humanoid-touch-dream.md`](../../wiki/entities/paper-humanoid-touch-dream.md) — canonical 论文实体页
- [`wiki/entities/htd-decoupled-wbc.md`](../../wiki/entities/htd-decoupled-wbc.md) — 解耦 WBC 组件、训练/蒸馏/部署
- [`wiki/methods/humanoid-transformer-touch-dreaming.md`](../../wiki/methods/humanoid-transformer-touch-dreaming.md) — HTD 策略方法页
- [`wiki/tasks/loco-manipulation.md`](../../wiki/tasks/loco-manipulation.md) — 接触丰富移动操作任务
