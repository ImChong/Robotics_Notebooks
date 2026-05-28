# open_duck_reference_motion_generator

> 来源归档

- **标题：** Open Duck Reference Motion Generator
- **类型：** repo
- **来源：** apirrone（Open Duck Project）
- **链接：** https://github.com/apirrone/Open_Duck_reference_motion_generator
- **入库日期：** 2026-05-28
- **一句话说明：** 基于 [Placo](https://github.com/Rhoban/placo) 的双足参数化步态引擎，批量生成 JSON 参考运动并拟合多项式系数，供 Playground 模仿奖励与 Isaac Gym（AWD）等训练栈使用。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-duck-reference-motion-generator.md`](../../wiki/entities/open-duck-reference-motion-generator.md)

---

## 依赖与安装

- **git-lfs** 必须安装（`sudo apt install git-lfs`），否则 STL 网格报错
- **uv** 包管理（与 Playground 一致）

## 支持机型（`--duck`）

- `go_bdx` — 接近 Disney BDX 比例
- `open_duck_mini` — v1 机体
- `open_duck_mini_v2` — 当前主线

步态参数范围见 `open_duck_reference_motion_generator/robots/<duck>/auto_gait.json`。

## 工作流

### 1. 生成参考运动

```bash
uv run scripts/auto_waddle.py -j8 --duck open_duck_mini_v2 --sweep
```

- `-jN`：并行 job 数
- `--sweep`：在 `auto_gait.json` 范围内枚举组合；`--num` 为随机采样
- 输出目录默认 `recordings/`（JSON）

### 2. 拟合多项式

```bash
uv run scripts/fit_poly.py --ref_motion recordings/
```

产出 **`polynomial_coefficients.pkl`** → 复制到 [Open_Duck_Playground](open_duck_playground.md) 的 `playground/open_duck_mini_v2/data/`

### 3. 可视化与调试

- `scripts/plot_poly_fit.py` — 检查拟合质量
- `scripts/replay_motion.py` — 回放单条 JSON
- `open_duck_reference_motion_generator/gait_playground.py` — 交互式步态调试

## 下游消费者

| 训练栈 | 链接 |
|--------|------|
| MuJoCo Playground（本项目主线） | [Open_Duck_Playground](open_duck_playground.md) |
| Isaac Gym | [rimim/AWD](https://github.com/rimim/AWD)、[SteveNguyen/openduckminiv2_playground](https://github.com/SteveNguyen/openduckminiv2_playground)（社区 fork） |

## 局限与 TODO（上游）

- 机器人描述拟拆为独立 submodule，避免版本重复
- 拟增加参考运动格式文档，便于从 mocap 转换
- 长期目标：泛化为任意双足的 Placo 步态生成器（如 Sigmaban）

## 与本仓库 wiki 的映射

- 实体页：`wiki/entities/open-duck-reference-motion-generator.md`
- 交叉：[imitation-learning.md](../../wiki/methods/imitation-learning.md)、[reward-design.md](../../wiki/concepts/reward-design.md)
