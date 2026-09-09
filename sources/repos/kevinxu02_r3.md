# KevinXu02/R3

> 来源归档

- **标题：** R³ — 3D Reconstruction via Relative Regression
- **类型：** repo
- **组织：** KevinXu02（Congrong Xu 等）
- **代码：** <https://github.com/KevinXu02/R3>
- **论文：** <https://arxiv.org/abs/2605.26519>
- **项目页：** <https://kevinxu02.github.io/r3-site/>
- **权重：** <https://huggingface.co/KevinXu02/R3>
- **许可证：** Apache-2.0（代码）；训练管线见 NOTICE（含上游非商业限制）
- **入库日期：** 2026-09-09
- **一句话说明：** DA3 骨干 + 成对相对位姿 MLP + 置信门控流式重建；`demo.py` 多模式（test/local/long/strided）+ Viser 可视化；训练代码已随 2026-06-19 发布。
- **沉淀到 wiki：** [`wiki/entities/paper-r3-relative-regression.md`](../../wiki/entities/paper-r3-relative-regression.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源** |
| **推理** | `demo.py` / `infer.py`；checkpoint `r3` / `r3_long` |
| **训练** | `R3/training/`（README 2026-06-19） |
| **评测** | **待发布**（README TODO） |
| **许可** | 代码 Apache-2.0；模型 CC BY-NC 4.0（DA3 衍生） |

## README 要点（2026-09-09）

### 安装

```bash
conda env create -f environment.yml
conda activate r3
pip install -e .
```

### Checkpoints

| 名称 | 训练视图 | 适用 |
|------|----------|------|
| `r3` | 4–32 | 室内 / 小范围（论文默认） |
| `r3_long` | 32–100 | 室外 / 长轨迹 |

### Demo 模式

| 模式 | 用途 |
|------|------|
| `test` | 冒烟：全 KV、无 fallback |
| `local` | 室内小范围 |
| `long` | 长轨迹 / 大室外 |
| `strided` | 时间抽帧视频 |

```bash
python demo.py --seq_path examples/indoor --no_viewer
python view.py --data_dir scratch/demo/<run_name>
```

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-r3-relative-regression.md`](../../wiki/entities/paper-r3-relative-regression.md)
- 项目页：[`sources/sites/r3-site.md`](../sites/r3-site.md)
- 论文摘录：[`sources/papers/r3_arxiv_2605_26519.md`](../papers/r3_arxiv_2605_26519.md)
