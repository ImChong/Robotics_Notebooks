# NavDP（InternRobotics/NavDP）

> 来源归档

- **标题：** NavDP
- **类型：** repo
- **来源：** Shanghai AI Laboratory / InternRobotics
- **链接：** <https://github.com/InternRobotics/NavDP>
- **项目页：** <https://wzcai99.github.io/navigation-diffusion-policy.github.io/>
- **论文：** <https://arxiv.org/abs/2505.08712>
- **许可：** 仓库 API 未识别 license；README 声明 open-sourced code 为 CC BY-NC-SA 4.0
- **入库日期：** 2026-07-28
- **一句话说明：** NavDP checkpoint 推理 server、IsaacSim / IsaacLab 异步评测 benchmark 与多导航 baseline 集成。
- **开源状态：** **已开源**；checkpoint 通过表单获取，原论文训练数据生成入口不是当前 README 主路径。
- **沉淀到 wiki：** [`paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md`](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 模型入口 | `baselines/navdp/navdp_server.py` |
| 环境 | Python 3.10；IsaacSim 4.2.0.2；IsaacLab 1.2.0 |
| 评测 | `eval_nogoal_wheeled.py` / `eval_pointgoal_wheeled.py` / `eval_imagegoal_wheeled.py` |
| 通信 | planner server 与 benchmark 通过 HTTP 解耦 |
| 执行 | benchmark 内 asynchronous MPC trajectory follower |
| 资产 | InternData-N1 / Scene-N1（Hugging Face） |

## 对 wiki 的映射

- [NavDP 论文实体](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)
- [NavDP 项目页](../sites/navdp.md)
- [论文 source](../papers/humanoid_pnb_navdp.md)
