# OmniContact_sim2sim（MuJoCo 部署与 sim2sim）

> 来源归档

- **标题：** OmniContact_sim2sim
- **类型：** repo
- **来源：** Noitom Robotics / HKUST 等（OmniContact 作者团队）
- **链接：** <https://github.com/Ingrid789/OmniContact_sim2sim>
- **项目页：** <https://omnicontact.github.io/>
- **论文：** [arXiv:2606.26201](https://arxiv.org/abs/2606.26201) — 归档见 [`sources/papers/omnicontact_arxiv_2606_26201.md`](../papers/omnicontact_arxiv_2606_26201.md)
- **入库日期：** 2026-07-01
- **一句话说明：** OmniContact 官方 **MuJoCo sim2sim** 栈：**CF-Gen** 任务空间参考生成 + **CF-Track** ONNX 低层策略；支持单 skill、skill chaining、NPZ 全轨迹跟踪与 Xbox 热切换 FSM 部署（镜像 sim2real 状态机）。
- **沉淀到 wiki：** [`wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md`](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)

---

## 核心定位

**OmniContact_sim2sim** 实现 [arXiv:2606.26201](https://arxiv.org/abs/2606.26201) 的 **Contact Flow 分层 loco-manipulation** 在 MuJoCo 上的闭环执行：

- **CFgen**：生成 carry / push / slide / relocate / kick 及链式 meta-skill 的 contact-flow 参考。
- **CFtrack**：跟踪 CFgen 参考或完整 `.npz` 人–物交互轨迹的低层 ONNX 策略。

两条执行路径：

| 脚本 | 用途 |
|------|------|
| `deploy_omnicontact/run_skill_omnicontact.py` | 脚本化直接运行（CFgen 或 NPZmotion） |
| `deploy_omnicontact/deploy_omnicontact.py` | Xbox 手柄热切换 FSM：`Passive → DefaultPose → LocoMode → OmniContact` |

---

## 环境与依赖

```bash
conda create -n omnicontact python=3.11 -y
conda activate omnicontact
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy onnx onnxruntime mujoco pyyaml scipy pygame
```

策略默认路径：`policy/omnicontact/model/policy.onnx`。

---

## 用法摘要

### CFgen 单 skill

```bash
python deploy_omnicontact/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy policy.onnx \
  --task carrybox \
  --init-pos 1.0 0.0 \
  --goal-pos 2.5 0.5
```

支持任务族：`loco`、`carrybox`、`pushbox*`、`slidebox*`、`relocateball`、`kickball` 等。

### Skill chaining

```bash
python deploy_omnicontact/run_skill_omnicontact.py \
  --reference-source CFgen \
  --policy policy.onnx \
  --task-chaining carry-push \
  --init-pos 1.0 0.0 \
  --goal-pos 2.5 0.5
```

链式 preset：`push-carry`、`carry-push`、`push-relocate`、`carry-carry`、`carry-carry-carry`、`carryheart` 等；各链对应独立 G1 MuJoCo XML 场景。

### NPZ 轨迹跟踪

```bash
python deploy_omnicontact/run_skill_omnicontact.py \
  --reference-source NPZmotion \
  --policy policy.onnx \
  --npz-dir data/relocateball/relocateball_motion_3_with_contact.npz
```

### 交互部署（Xbox）

| 输入 | 状态切换 |
|------|----------|
| `START` | DefaultPose |
| `R1 + A` | LocoMode |
| `L1 + A` | OmniContact policy |
| `L3` | Passive |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [Humanoid-Gym](./humanoid-gym.md) | 同类 Isaac Gym→MuJoCo sim2sim 校验文化 |
| [VisualMimic](./visualmimic.md) | 分层 loco-manip + sim2sim 推理对照 |
| [ProtoMotions](./protomotions.md) | 多后端 sim2sim 与观测对齐叙事 |
| [Unitree G1](../../wiki/entities/unitree-g1.md) | 默认人形平台 |

## 对 wiki 的映射

- 实体页：[OmniContact（论文）](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)
- 数据集：[omnicontact-dataset.md](../datasets/omnicontact-dataset.md)
- [Sim2Real](../../wiki/concepts/sim2real.md) — sim2sim 作为迁移前验门
