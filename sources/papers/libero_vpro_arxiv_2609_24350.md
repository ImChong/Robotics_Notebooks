# LIBERO-VPro（arXiv:2609.24350）

> 来源归档（paper）

- **标题：** LIBERO-VPro: Benchmarking Closed-Loop Visual Robustness of Robotic Foundation Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24350>
- **PDF：** <https://arxiv.org/pdf/2609.24350>
- **项目页：** <https://huiqiongli.github.io/LIBERO-VPro/>
- **代码 / 数据：** <https://github.com/huiqiongli/LIBERO-VPro-Dataset>
- **入库日期：** 2026-10-18
- **最后核对：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 全文通读 `arXiv:2609.24350`（含 Table II 闭环鲁棒性对比、A1/A2 空间先验诊断、LingBot-VA 递归 rollout 分析、200 次 Franka FR3 真机验证）并核对公开仓库 `huiqiongli/LIBERO-VPro-Dataset`（`scripts/run_eval.py`、`runnable_settings/run_setting.py`、`PAPER_FACING_SETTINGS.json`）
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 在执行时扰动视觉证据（缺失、延迟、多视角矛盾、场景变化），做 LIBERO 上的闭环视觉鲁棒性基准。

## 开源状态

- **代码与评测资产已部分开源**（2026-11-21 复核）：`huiqiongli/LIBERO-VPro-Dataset` 已发布（代码 MIT、数据集资产 CC BY 4.0），包含 `scripts/run_eval.py`、`PAPER_FACING_SETTINGS.json` 评测配置与 FastWAM 集成示例；OpenVLA-OFT / $\pi_0$ / $\pi_{0.5}$ / LaWAM 的模型服务与权重仍需由外部运行器挂载。

## 核心摘录（已核验论文原文与官方仓库）

- Table II 对比 6 类策略从干净环境到 11 类扰动平均成功率（Avg SR）：OpenVLA-OFT `97.0 -> 56.40`、$\pi_0$ `92.5 -> 60.55`、$\pi_{0.5}$ `97.2 -> 64.49`、FastWAM `98.8 -> 66.00`、LingBot-VA `97.0 -> 67.31`、LaWAM `98.5 -> 65.68`。
- Same-Position Distractor 致多数模型崩塌（仅 `2.5%–22.25%`）；LingBot-VA 在递归 rollout 下由 `97.0%` 降至 `53.8%`（Long 套件 `98.0% -> 22.0%`）。
- A1/A2 空间先验遮挡诊断实验显示多项测试在物体被遮挡后仍保持 `>80%–96%` 成功率，揭示对物体绝对位置先验的依赖；另含 200 次 Franka FR3 真机验证。

## 对 wiki 的映射

- [paper-libero-vpro](../../wiki/entities/paper-libero-vpro.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
