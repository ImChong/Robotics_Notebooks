# ArtGym（ArtManip 官方实现）

> 来源归档

- **标题：** ArtGym
- **类型：** repo
- **来源：** ArtManip 作者团队（GitHub `youngcv/artgym`）
- **链接：** <https://github.com/youngcv/artgym>
- **配套论文：** [ArtManip（arXiv:2609.12498）](https://arxiv.org/abs/2609.12498)
- **配套项目页：** <https://artmanip.github.io/>
- **入库日期：** 2026-09-15
- **最近复核：** 2026-09-15
- **一句话说明：** ArtManip 官方 Isaac Gym 仿真栈：**铰接物体抓取验证、Teacher（特权 LSTM-PPO + SAPG）、Student 蒸馏（proprio-only + TCN 历史）** 与连续开合评测；依赖子模块 `make_data`（程序化物体）与 `func_lygra`（功能抓取合成）。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-artmanip.md`](../../wiki/entities/paper-artmanip.md)

---

## 核心定位

**ArtGym** 是 [ArtManip](https://arxiv.org/abs/2609.12498) 的 **官方仿真与训练代码**。目标是在 **Sharpa** 灵巧手平台上学习 **类别级铰接物体 in-hand 操作**：策略既要 **维持对自由漂浮物体的抓取稳定**，又要 **驱动内部 prismatic/revolute 关节** 完成开合等任务。

---

## 安装（README / install.md 摘录）

- **推荐环境：** Linux、RTX 4090、Python 3.8、CUDA 11.8 PyTorch 2.1。
- **仿真后端：** [IsaacGym_TacSL](https://github.com/)（README 通过 gdown 拉取预览包）。
- **子模块（`--recursive` 克隆）：**
  - `make_data/` — 生成 `./assets/objects/...`
  - `func_lygra/` — 生成 `./caches/initial_grasp/...`

```bash
git clone --recursive https://github.com/youngcv/artgym.git
cd artgym
# 见 install.md：conda env、PyTorch、IsaacGym_TacSL、hydra 等依赖
```

---

## 主工作流（README Pipeline）

| 阶段 | 入口 | 作用 |
|------|------|------|
| 0. 资产与初始抓取 | `make_data`、`func_lygra` | 程序化铰接物体 + 功能抓取缓存 |
| 1. 验证抓取 | `scripts/validate_all_instances.sh` / `isaacgymenvs.valid_grasp` | 仿真中筛 **稳定初始抓取**（位姿/旋转阈值） |
| 2. 训练 Teacher | `python -m isaacgymenvs.train task=artmanip train=artmanipSAPGPrivLSTMPPO` | 特权状态 PPO；`task.env.graspSplit=valid` |
| 3. 评测 Teacher | `isaacgymenvs.eval_consecutive` | 连续开合周期成功率；可导出 `success` grasp pool |
| 4. Teacher 推理 | `isaacgymenvs.infer teacher` | 选定抓取 + 保存 `cur_targets.npy` / 视频 |
| 5. 蒸馏 Student | `isaacgymenvs.distill` | proprio-only + `--custom_tcn` 历史；余弦蒸馏损失 |
| 6. 评测 Student | `isaacgymenvs.eval_consecutive --student-artifact ...` | 部署侧策略连续成功评测 |

**典型配置示例：** `hand=sharpa`、`object=knife`、`asset_dir=knife_30`；Teacher 默认 `numEnvs=16000`（每 GPU）。

---

## 开源边界（截至 2026-09-15）

| 项 | 状态 |
|----|------|
| 仿真环境 + 训练/蒸馏脚本 | **已发布**（本仓库） |
| 程序化物体与抓取生成 | **已发布**（子模块） |
| 预训练 checkpoint / 真机部署栈 | **以仓库 Issue / 更新为准**；论文报 12 真机零样本，代码侧重 sim 管线 |

---

## 对 wiki 的映射

- [ArtManip（论文实体）](../../wiki/entities/paper-artmanip.md)
- 项目页：[`sources/sites/artmanip-github-io.md`](../sites/artmanip-github-io.md)
