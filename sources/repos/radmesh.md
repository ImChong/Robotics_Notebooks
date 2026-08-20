# threedle/radmesh

- **标题：** RADmesh 官方实现（ECCV 2026 Oral）
- **类型：** repo
- **URL：** <https://github.com/threedle/radmesh>
- **许可：** 仓内 README 未列 SPDX；以仓库为准
- **配套论文：** [arXiv:2608.17182](https://arxiv.org/abs/2608.17182) — [`sources/papers/radmesh_arxiv_2608_17182.md`](../papers/radmesh_arxiv_2608_17182.md)
- **项目页：** [`sources/sites/radmesh-threedle-github-io.md`](../sites/radmesh-threedle-github-io.md)
- **入库日期：** 2026-08-20

## 一句话说明

文本引导 **remesh-aware** 网格形变：配置 JSON + `run_optimization.py`；几何核 `radmesh/deformations.py`（libigl + cholespy）；CSD + DeepFloyd IF + nvdiffrast。

## 仓库状态（2026-08-20 核查）

| 项 | 内容 |
|----|------|
| 入口 | `run_optimization.py`（`-c` JSON 配置） |
| 示例 | `example-config-localized.json`、`example-config-wholemesh.json`、`example-run/` |
| 几何 | `radmesh/deformations.py`；vendored `radmesh/pytorch3d/` |
| 环境 | `env_setup.sh`；Python 3.10–3.12；CUDA + HF DeepFloyd IF |
| 可视化 | `thlog replay psrec-*.npz`；`view_drmsh_npz.py drmsh-*.npz` |
| Headless | 环境变量 `NO_POLYSCOPE=1` |

最短复现：conda 环境 → `env_setup.sh` → HF 登录下载 IF → `python run_optimization.py -c example-config-localized.json`。

## 与 wiki 的关系

- 实体页：[paper-radmesh](../../wiki/entities/paper-radmesh.md) — 含源码运行时序图。
