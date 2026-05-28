---
type: entity
tags: [imitation, locomotion, placo, reference-motion, open-source, biped]
status: complete
updated: 2026-05-28
related:
  - ./open-duck-mini.md
  - ./open-duck-playground.md
  - ../methods/imitation-learning.md
  - ../concepts/reward-design.md
  - ../methods/disney-olaf-character-robot.md
sources:
  - ../../sources/repos/open_duck_reference_motion_generator.md
summary: "Open Duck Reference Motion Generator 用 Rhoban Placo 参数化生成双足参考步态，批量 sweep 后拟合多项式系数，供 Playground 模仿奖励与 Isaac Gym 训练栈消费。"
---

# Open Duck Reference Motion Generator

**Open Duck Reference Motion Generator** 为 Open Duck 生态提供 **可批量再生的参考运动**：基于 [Placo](https://github.com/Rhoban/placo) 的自动步态引擎生成 JSON 轨迹，再拟合为 `polynomial_coefficients.pkl`，供 RL 模仿奖励读取。

## 为什么重要

- **解耦「步态设计」与「策略优化」：** 研究者可在 `auto_gait.json` 里 sweep 步态参数，而不必手调 mocap 或动画曲线。
- **多后端一致：** 同一套运动可服务 **MuJoCo Playground**（主线）与 **Isaac Gym / AWD**（早期或社区 fork）。
- **贴近 Disney BDX 管线：** 与 BDX 论文中「参数化参考 + RL 跟踪」的工程思路一致，便于与 [Disney Olaf](../methods/disney-olaf-character-robot.md) 等娱乐双足对照。

## 核心结构/机制

1. **`auto_waddle.py`** — 按 `--duck`（`open_duck_mini_v2` 等）与 `--sweep` / `--num` 生成 `recordings/*.json`
2. **`fit_poly.py`** — 将轨迹拟合为多项式系数 → **`polynomial_coefficients.pkl`**
3. **`gait_playground.py`** — 交互调试步态参数
4. **机器人描述** — `robots/<duck>/auto_gait.json` 定义 sweep 范围

**典型流水线：**

```bash
uv run scripts/auto_waddle.py -j8 --duck open_duck_mini_v2 --sweep
uv run scripts/fit_poly.py --ref_motion recordings/
# 复制 polynomial_coefficients.pkl → Open_Duck_Playground/.../data/
```

## 常见误区或局限

- **必须 git-lfs：** STL 网格未拉取会导致 Placo 加载失败（`git lfs pull`）。
- **格式文档缺失：** 从 mocap 转换需自行对齐 JSON 字段；上游 TODO 中计划补规范。
- **机器人描述重复：** 未来拟拆 submodule，当前与 Playground MJCF 需手动保持版本一致。

## 参考来源

- [sources/repos/open_duck_reference_motion_generator.md](../../sources/repos/open_duck_reference_motion_generator.md)
- [apirrone/Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator)

## 关联页面

- [Open Duck Mini](./open-duck-mini.md)
- [Open Duck Playground](./open-duck-playground.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Reward Design](../concepts/reward-design.md)

## 推荐继续阅读

- [Rhoban/placo](https://github.com/Rhoban/placo) — 逆运动学与步态引擎
- [Peng et al., DeepMimic](https://arxiv.org/abs/1804.06401) — 物理角色模仿 RL 经典参照
