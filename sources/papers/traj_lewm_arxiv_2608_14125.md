# Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost（arXiv:2608.14125）

> 来源归档（ingest）

- **标题：** Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost
- **类型：** paper / JEPA / latent world model / goal-conditioned planning / trajectory preference
- **arXiv：** <https://arxiv.org/abs/2608.14125>
- **代码：** <https://github.com/XiaodiHuang-code/Traj_LeWM>（MIT；源码发布，无预置 checkpoint）
- **作者：** Xiaodi Huang\*、Ziyi Ding\*、Jingtian Wan、Yuchen Liu、Yuan Zhang、Xiao-Ping Zhang、Jiayu Chen†、Zhang Zhang†、Tao Huang†
- **机构：** 中国科学院自动化研究所；上海交通大学；清华大学深圳国际研究生院；香港大学；INFIFORCE；中国科学技术大学；北京大学
- **入库日期：** 2026-09-09
- **一句话说明：** 在 [LeWM](../../wiki/entities/paper-lewm.md) 上保留下一步预测 + 终点距离，新增目标条件 **Latent Trajectory Cost (LTC)**：训练期用轨迹偏好塑形表征，规划期与终点分联合排序；四仿真任务相对 LeWM **+3/+14/+7/+7 pp**，Franka FR3 真机 **50%→70%**。

## 开源状态（仓库核查，2026-09-09）

- **已开源（代码）：** [`XiaodiHuang-code/Traj_LeWM`](https://github.com/XiaodiHuang-code/Traj_LeWM) 为 source-only release：含 `train.py` / `eval.py` / `jepa.py`、四环境配置与分析脚本；README 声明不含 checkpoint。

## 核心论文摘录（MVP）

### 1) 动机：终点不足以区分候选

- **链接：** §1；Fig. 1
- **摘录要点：** LeWM 仅用预测终点到目标的距离排序 CEM 候选；同起终点、相近终点成本的轨迹执行结果可不同。下一步损失 + SIGReg 未直接监督「完整轨迹相对目标的质量」。
- **对 wiki 的映射：** [paper-lewm](../../wiki/entities/paper-lewm.md)、[paper-traj-lewm](../../wiki/entities/paper-traj-lewm.md)

### 2) LTC 与轨迹偏好学习

- **链接：** §3
- **摘录要点：** LTC 将目标相对完整 latent 轨迹映射为标量代价；偏好来自 (i) 目标错配专家轨迹、(ii) 终点保持的 latent 扰动负样本、(iii) 终点-only 闭环失败挖掘。保留 LeWM 的 \(L_{\text{pred}}+\text{SIGReg}\)。
- **对 wiki 的映射：** [paper-traj-lewm](../../wiki/entities/paper-traj-lewm.md)

### 3) 主结果与真机

- **链接：** Tab. 2；§4
- **摘录要点：** Push-T / OGBench-Cube / Reacher / Two-Room 成功率 **99/88/93/94%**（三 seed 均值），相对 LeWM **96/74/86/87%** 提升 **3/14/7/7 pp**；20 项 Franka FR3 任务 **50%→70%**。LTC 参数 <1M。
- **对 wiki 的映射：**
  - [paper-traj-lewm](../../wiki/entities/paper-traj-lewm.md)
  - [paper-state-readout-decoupling](../../wiki/entities/paper-state-readout-decoupling.md) — 同组 latent 规划线

## BibTeX

```bibtex
@article{huang2026trajlewm,
  title   = {Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost},
  author  = {Huang, Xiaodi and Ding, Ziyi and Wan, Jingtian and Liu, Yuchen and Zhang, Yuan and Zhang, Xiao-Ping and Chen, Jiayu and Zhang, Zhang and Huang, Tao},
  journal = {arXiv preprint arXiv:2608.14125},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-traj-lewm.md`](../../wiki/entities/paper-traj-lewm.md)
- 代码归档：[`sources/repos/traj-lewm.md`](../repos/traj-lewm.md)
