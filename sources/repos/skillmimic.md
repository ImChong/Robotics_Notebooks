# SkillMimic（wyhuai/SkillMimic）

> 来源归档

- **标题：** SkillMimic
- **类型：** repo
- **来源：** Yinhuai Wang et al.（HKUST / Unitree / PKU / Tsinghua / IDEA / Tencent / CMU）
- **链接：** <https://github.com/wyhuai/SkillMimic>
- **项目页：** <https://ingrid789.github.io/SkillMimic/>
- **论文：** [arXiv:2408.15270](https://arxiv.org/abs/2408.15270) — 归档见 [`sources/papers/skillmimic_arxiv_2408_15270.md`](../papers/skillmimic_arxiv_2408_15270.md)
- **许可：** Apache-2.0
- **入库日期：** 2026-07-28
- **一句话说明：** CVPR 2025 Highlight 官方仓：**训练与评测代码、预训练 LLC/HLC、BallPlay-M 子集、Blender 渲染** 已发布；完整原始 BallPlay-M 与数据处理仍 TODO。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md`](../../wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md)

---

## 核心定位

在 **Isaac Gym** 上复现 SkillMimic：**统一 HOI 模仿** 训多技能 low-level policy，再训 **HLC** 组合技能完成 heading / circling / scoring / throwing。实现基于 **ASE** 与 **PhysHOI**。

---

## 发布进度（README TODOs，截至 2026-07-28）

| 组件 | 状态 |
|------|------|
| 训练与评测代码 | ✅ |
| BallPlay-M 子集 | ✅ `skillmimic/data/motions/BallPlay-M/` |
| Blender 渲染 | ✅ `blender_for_SkillMimic/` |
| 预训练模型 | ✅ `skillmimic/data/models/`（含 `mixedskills` LLC 与各 HLC） |
| 完整原始 BallPlay-M + 数据处理代码 | ⬜ TODO |

---

## 安装与主入口

- **环境：** `conda create -n skillmimic python=3.8` + `pip install -r requirements.txt`，或 `conda env create -f environment.yml`
- **依赖：** 需自行下载并 `pip install -e` **Isaac Gym Preview 4**
- **统一入口：** `python skillmimic/run.py ...`

### Skill Policy（LLC）

```bash
# 推理（键盘切换技能）
python skillmimic/run.py --test --task SkillMimicBallPlay --num_envs 16 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/layup \
  --checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --state_init 20 --episode_length 140

# 训练
python skillmimic/run.py --task SkillMimicBallPlay \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/layup --headless
```

### High-Level Controller

- 任务类：`HRLCircling`、`HRLHeadingEasy`、`HRLThrowing`、`HRLScoringLayup`
- 配置：`skillmimic_hlc.yaml` + `hrl_humanoid_discrete_*.yaml`
- 需 `--llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth`

---

## 关键目录（对齐时序图节点）

| 路径 | 角色 |
|------|------|
| `skillmimic/run.py` | CLI 训练 / 测试入口 |
| `skillmimic/env/tasks/skillmimic.py` | BallPlay skill policy 环境 |
| `skillmimic/env/tasks/hrl_*.py` | HLC 任务环境 |
| `skillmimic/data/cfg/` | env / rlg 训练配置 |
| `skillmimic/data/motions/BallPlay-M/` | 运动子集 |
| `skillmimic/data/models/` | 预训练 LLC / HLC |
| `blender_for_SkillMimic/` | 渲染 |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [ASE](../papers/ase.md) | 代码与技能嵌入族基座 |
| [Learning to Ball](./humanoid_pnb_learning-to-ball.md)（PNB） | 同主题篮球长程组合对照 |
| PhysHOI / SkillMimic-V2（外链） | 同作者 HOI 模仿前后作 |

## 对 wiki 的映射

- 实体页：[SkillMimic](../../wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md)
- 方法页：[ASE](../../wiki/methods/ase.md)、[Hierarchical RL](../../wiki/methods/hierarchical-reinforcement-learning.md)
