# muninn（UIUC · 轨迹扩散免训练加速）

> 来源归档

- **标题：** Muninn
- **类型：** repo
- **来源：** Gokul Puthumanaillam 等（UIUC Ornik 组主导）
- **链接：** <https://github.com/gokulp01/Muninn>
- **License：** 以仓库为准（Diffuser 集成为 MIT）
- **Stars：** ~6（2026-07，以 GitHub 为准）
- **入库日期：** 2026-07-16
- **一句话说明：** RSS 2026 论文 **Muninn** 官方实现——纯 Python **`muninn`** 核心包 + **[Diffuser](https://github.com/jannerm/diffuser)** 集成；提供 `DiffusionAdapter` 接口、split-conformal 标定与预算式缓存采样，D4RL HalfCheetah / Walker2d 一键 **calibrate → certify → evaluate** 管线。
- **沉淀到 wiki：** [`wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md`](../../wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md)

---

## 核心定位

**Muninn** 是 arXiv:2605.09999 / RSS 2026 的 **模型无关推理加速层**：包装已有轨迹扩散规划器或扩散策略，**不重训** denoiser，通过 probe + 解析灵敏度 + conformal 预算选择性 **跳过昂贵 forward**。

```
muninn/                        # 核心库（model-agnostic）
├── adapter.py                 # DiffusionAdapter 五方法接口
├── sampler.py                 # Algorithm 1 预算式缓存循环
├── calibration.py             # split-conformal 逐步误差上界
└── coefficients.py            # DDPM 解析灵敏度 L_t, K_t
integrations/
└── diffuser/                  # Diffuser 基准复现
    ├── muninn_diffuser/       # DiffuserAdapter + MuninnGuidedPolicy
    └── scripts/muninn_*.py    # test / calibrate / eta_probe / eval / report
```

---

## 功能摘要（README）

| 维度 | 内容 |
|------|------|
| 核心依赖 | `numpy`, `torch`, `scikit-learn`（`uv pip install -e .`） |
| Diffuser 基准 | 需 MuJoCo 2.1、`setup_env.sh`、CUDA torch、D4RL、预训练 checkpoint |
| 数据集 | `halfcheetah-medium-v2`, `walker2d-medium-v2`（可换 `--dataset`） |
| 管线 | `muninn_test` → `muninn_calibrate` → `muninn_eta_probe` → `muninn_eval` → `muninn_report` |
| 关键旋钮 | `--eta` 偏差预算、`--alpha` 风险、`--prefix_forbid` / `--suffix_forbid` 禁复用区 |
| 集成方式 | 实现 `muninn.DiffusionAdapter` 五方法：`probe`, `compute`, `posterior`, `alphas_cumprod`, `sample_shape` |

---

## 快速上手（Diffuser 基准）

```bash
cd integrations/diffuser
bash setup_env.sh && source env.sh
# 预训练权重 -> logs/pretrained（见 README_DIFFUSER.md）

python scripts/muninn_calibrate.py --dataset halfcheetah-medium-v2 --logbase logs/pretrained --vis_freq -1
python scripts/muninn_eta_probe.py  --dataset halfcheetah-medium-v2 --logbase logs/pretrained --vis_freq -1
python scripts/muninn_eval.py       --dataset halfcheetah-medium-v2 --logbase logs/pretrained --mode muninn --eta 4.2
```

---

## 自定义扩散模型集成要点

- **`probe`**：denoiser 最便宜前缀（stem / 首块 attention），目标 ≪ 完整 forward。
- **`compute`**：denoiser + guidance 一并计入；reuse 步 **整段跳过**。
- **`gamma`**：轨迹偏差度量常数；默认 $d = \|\cdot\|_F / \sqrt{H}$。
- 完整范例：`integrations/diffuser/muninn_diffuser/adapter.py`（含 value guidance 与 inpainting）。

---

## 与相邻工具的关系

- **后端**：[Diffuser](https://github.com/jannerm/diffuser)（Janner et al. 2022）作为 D4RL 价值引导规划基准。
- **对照加速**：FewSteps、FixedSkip、蒸馏 one-step、LearnedExit — Muninn 强调 **免训练 + 可证偏差预算**。
- **下游策略**：[Diffusion Policy](https://github.com/real-stanford/diffusion_policy) 类 visuomotor 在论文 Table III 有评测，仓库当前以 Diffuser 集成为主。

---

## 参考来源

- [muninn_arxiv_2605_09999.md](../papers/muninn_arxiv_2605_09999.md)
- <https://github.com/gokulp01/Muninn>
- <https://arxiv.org/abs/2605.09999>
