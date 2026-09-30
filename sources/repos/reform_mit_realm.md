# MIT-REALM/reform

> 来源归档

- **标题：** ReFORM 官方实现（Jax）
- **类型：** repo
- **作者：** Songyuan Zhang, Oswin So, H. M. Sabbir Ahmad, Eric Yang Yu, Matthew Cleaveland, Mitchell Black, Chuchu Fan（MIT / Boston University / MIT Lincoln Laboratory）
- **代码：** <https://github.com/MIT-REALM/reform>
- **Stars / Forks：** ~9 / 0（2026-09-30 快照）
- **论文：** <https://openreview.net/forum?id=YvFsyRReeN>（ICLR 2026）
- **项目页：** <https://mit-realm.github.io/reform/>
- **入库日期：** 2026-09-30
- **一句话说明：** ICLR 2026 **ReFORM** 与 **FQL / IFQL / DSRL** 的 Jax 训练栈；`scripts/train.py` + `reform/agents/reform.py`，OGBench **singletask** 环境。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-reform-iclr-2026.md`](../../wiki/entities/paper-reform-iclr-2026.md)

## 仓库结构（顶层）

```
reform/
├── reform/
│   ├── agents/          # reform, fql, ifql, dsrl + module (actor, value, distribution…)
│   ├── env/             # OGBench 环境封装
│   ├── trainer/         # trainer.py, datasets.py
│   └── utils/
├── scripts/
│   ├── train.py         # python scripts/train.py <algo> --env-name …
│   └── test.py          # 评估 logs/ 下 checkpoint
├── requirements.txt
└── setup.py
```

## 快速复现（README）

```bash
conda create -n reform python=3.12 && conda activate reform
pip install -r requirements.txt && pip install -e .
python scripts/train.py reform --env-name cube-single-noisy-singletask-task1-v0 --steps 3000000 --seed 0
python scripts/test.py --path ./logs/cube-single-noisy-singletask-task1-v0/reform/seed0_xxxxxxxxxx
```

- 环境须使用 OGBench 的 **`singletask`** 命名（非 goal-conditioned）。
- 论文主结果命令列表见 README `<details>`（antmaze / cube / scene 的 clean & noisy）。

## 对 wiki 的映射

- [ReFORM 实体页](../../wiki/entities/paper-reform-iclr-2026.md) — 方法归纳与 OGBench 读法
- [sources/papers/reform_iclr_2026.md](../papers/reform_iclr_2026.md)
- [sources/sites/reform-mit-realm-github-io.md](../sites/reform-mit-realm-github-io.md)
