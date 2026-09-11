# JEPA Policy

> 来源归档

- **标题：** JEPA Policy
- **类型：** repo
- **链接：** <https://github.com/jiejie567/JEPA-Policy>
- **论文：** <https://arxiv.org/abs/2609.09630>
- **项目页：** <https://jiejie567.github.io/JEPA-Policy/>
- **许可：** MIT
- **基座：** [Minimum Flow Policies](https://github.com/simchowitzlabpublic/much-ado-about-noising)（MIP）
- **入库日期：** 2026-09-11
- **再核日期：** 2026-09-11
- **一句话说明：** 扩散-free 模仿学习：共享 Transformer 联合预测动作块与未来视觉表征；支持 robomimic / LIBERO / MimicGen；含 ARX5/X5 真机推理栈（权重未入库）。
- **沉淀到 wiki：** [`wiki/entities/paper-jepa-policy.md`](../../wiki/entities/paper-jepa-policy.md)

---

## 仓库入口（README，2026-09-11）

| 组件 | 说明 |
|------|------|
| 安装 | Python 3.12；`uv sync --extra dev` |
| 仿真训练 | `uv run examples/train_robomimic.py -cn exps/jepa_policy.yaml task=square_ph_image` |
| 基线 | `exps/mip_action_only.yaml`（动作-only MIP）；`baselines/` 内 Diffusion Policy 配方 |
| 评测 | `mode=eval optimization.model_path=/path/to/checkpoint.pt` |
| 真机 | `real_robot/README.md` — 五任务 launcher、安全与 dry-run；checkpoint 见 75-checkpoint manifest（未随 Git） |
| 测试 | `pytest tests/test_joint_action_future_mip.py` 等 |

---

## 方法配置（公开 preset）

- robomimic/MimicGen 动作 horizon **10**；LIBERO **16**
- MIP **两步**预测，插值时间 **0.9**
- 未来表征 token **1** 个，future horizon **4**
- adaptive future-loss ratio **0.1**
- 当前与未来观测 **共享 temporal crop**

---

## 开源边界

| 已发布 | 未随 Git |
|--------|----------|
| 仿真训练/评测 Hydra 配置 | 真机 checkpoint |
| `real_robot/` 推理与安全栈 | 数据集、相机序列号 |
| 基线 Diffusion Policy overlay | 内部机器路径 |
