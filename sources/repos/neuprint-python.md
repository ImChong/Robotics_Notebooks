# neuprint-python

> 来源归档

- **标题：** neuprint-python
- **类型：** repo
- **机构：** connectome-neuprint / HHMI Janelia 生态
- **链接：** https://github.com/connectome-neuprint/neuprint-python
- **文档：** https://connectome-neuprint.github.io/neuprint-python/docs/
- **PyPI：** `neuprint-python`
- **Stars：** ~43+（2026-09）
- **入库日期：** 2026-09-10
- **一句话说明：** neuPrint 连接组分析服务的官方 Python 客户端，提供神经元检索、邻接图与 ROI 统计等 API。
- **代码：** https://github.com/connectome-neuprint/neuprint-python（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/neuprint.md`](../../wiki/entities/neuprint.md)（与 neuPrint 服务合并叙述）

---

## 安装

```bash
pip install neuprint-python
# 或
conda install -c flyem-forge neuprint-python
```

依赖建议：`pyarrow>=20`、`numpy>=2`、`pandas>=2`（pixi/conda 环境见 README）。

---

## 典型 API

| 函数 | 用途 |
|------|------|
| `Client(url, dataset, token)` | 连接 neuPrint 实例 |
| `fetch_neurons(criteria)` | 按类型/ID 取神经元元数据 |
| `fetch_adjacencies(sources, targets)` | 取突触邻接边 |
| `fetch_roi_names()` / ROI 统计 | 神经纤维网室分布 |

Male CNS 数据集名：`male-cns:v1.0`。

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 客户端 | **已开源** |
| 运行前提 | neuPrint 账号 + API token；服务由 Janelia 托管 |

---

## 对 wiki 的映射

- [neuPrint](../../wiki/entities/neuprint.md)
- [Male CNS Connectome](../../wiki/entities/male-cns-connectome.md)
