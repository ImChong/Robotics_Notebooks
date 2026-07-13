# MoRE（TeleHuman 官方实现）

- **标题：** MoRE — Mixture of Residual Experts for Humanoid Lifelike Gaits
- **类型：** repo
- **仓库：** <https://github.com/TeleHuman/MoRE>
- **项目页：** <https://more-humanoid.github.io/>
- **论文：** arXiv:[2506.08840](https://arxiv.org/abs/2506.08840)
- **机构：** 中国科学技术大学（USTC）、中国电信人工智能研究院（TeleAI）、哈尔滨工程大学（HEU）、上海科技大学（ShanghaiTech）
- **硬件：** Unitree G1（16 DoF 腿+臂）+ Intel RealSense D435i
- **收录日期：** 2026-07-13
- **许可：** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)

## 一句话摘要

TeleHuman 开源的 **MoRE 两阶段训练栈**：Isaac Gym 上先训 **深度条件 base locomotion**（`g1_16dof_loco`），再加载 checkpoint 训 **latent residual MoE + 多判别器 AMP**（`g1_16dof_resi_moe`）；含 MuJoCo sim2sim 四类地形验证与 gait command 键盘切换（Walk-Run / High-Knees / Squat）。

## 为何值得保留

- **AMP 专题 #08 官方复现入口：** 与 [MoRE 实体页](../../wiki/entities/paper-amp-survey-08-more.md) 论文归纳一一对应，补齐训练命令、迭代数与环境规模等工程细节。
- **两阶段管线可拆解：** Stage 1 与 Stage 2 为独立 `legged_gym` task，便于只复现 base 穿越或完整 MoRE 步态切换。
- **部署适配细节：** Stage 2 末 10k iter 启用 **body mask** 做部署适配；需从 OneDrive 下载 `body_mask_data`。
- **社区参考：** GitHub 280+ stars（2026-07）；与 [PBHC](./pbhc.md)、[HumanoidSoccer](./humanoid_soccer.md) 等同属 TeleHuman 人形 RL 谱系。

## 环境与依赖

| 组件 | 版本 / 说明 |
|------|-------------|
| Python | conda `more` 环境（`conda_env.yml`） |
| PyTorch | 2.3.1 + CUDA 12.1 |
| 仿真 | NVIDIA Isaac Gym + **rsl_rl** + 本仓库 `legged_gym` |
| 验证 | MuJoCo（`deploy/deploy_mujoco/`） |

## 训练流程（编译自 README）

### Stage 1 — Base locomotion

```bash
python legged_gym/scripts/train.py --task g1_16dof_loco --headless
```

- 推荐 **≥40k** iter（README 建议 30k–50k）；**≥3000** 并行环境。
- 产出 checkpoint 供 Stage 2 加载（配置项 `loco_expert_ckpt_path`）。

### Stage 2 — Residual MoE + AMP

1. 在 `g1_16dof_moe_residual_config.py` 设置 base policy 路径。
2. 下载 [body mask 数据集](https://1drv.ms/u/c/ec72522c19d152ff/EQTi52kL1hNOg43MWMr_1qkBoimXUGg-4a1-HY-f0YIYIw?e=TR5uBE) 至 `./body_mask_data`。
3. 训练：

```bash
python legged_gym/scripts/train.py --task g1_16dof_resi_moe --headless
```

- 共 **40k** iter：前 30k 训残差网络，后 10k 启用 body mask。
- 推荐 **≥6000** 环境；支持 `torchrun` 多 GPU。

### 可视化与 gait 切换

```bash
python legged_gym/scripts/play.py --task g1_16dof_loco/g1_16dof_resi_moe --load_run ${policy_path}
```

| 按键 | 功能 |
|------|------|
| `Z` / `X` / `C` | gait command：Walk-Run / High-Knees / Squat |
| `W A S D` | 速度命令 |
| `Space` | 暂停仿真 |

### MuJoCo 验证

四类地形：**Roughness / Pit / Stairs / Gap**；通过 `g1_16dof_resi_moe.yaml` 选择场景：

```bash
python deploy/deploy_mujoco/deploy_mujoco_with_resi.py g1_16dof_resi_moe.yaml
```

## 仓库结构（要点）

| 路径 | 作用 |
|------|------|
| `legged_gym/` | Isaac Gym 环境与 `g1_16dof_loco` / `g1_16dof_resi_moe` 任务 |
| `rsl_rl/` | PPO 训练库（可编辑安装） |
| `deploy/deploy_mujoco/` | MuJoCo sim2sim 脚本与 YAML |
| `body_mask_data/` | Stage 2 部署适配用 body mask（需外部下载） |
| `docs/` | 方法示意图与地形预览图 |

## 对 Wiki 的映射

- [MoRE（AMP 专题 #08）](../../wiki/entities/paper-amp-survey-08-more.md) — 方法归纳与文献对照
- [more_mixture_residual_experts_arxiv_2506_08840.md](../papers/more_mixture_residual_experts_arxiv_2506_08840.md) — 论文 source
- [amp-reward.md](../../wiki/methods/amp-reward.md)、[terrain-adaptation.md](../../wiki/concepts/terrain-adaptation.md)、[locomotion.md](../../wiki/tasks/locomotion.md)
- [unitree-g1.md](../../wiki/entities/unitree-g1.md)、[lafan1-dataset.md](../../wiki/entities/lafan1-dataset.md)
- [PBHC](./pbhc.md) — 同 TeleHuman 人形高动态 / 跟踪栈

## 参考来源（原始）

- 代码仓库：<https://github.com/TeleHuman/MoRE>
- 项目页：<https://more-humanoid.github.io/>
- arXiv:2506.08840
