# neuPrint

> 来源归档

- **标题：** neuPrint
- **类型：** site
- **机构：** HHMI Janelia
- **链接：** https://neuprint.janelia.org/
- **Python 客户端：** https://github.com/connectome-neuprint/neuprint-python
- **文档：** https://connectome-neuprint.github.io/neuprint-python/docs/
- **入库日期：** 2026-09-10
- **一句话说明：** Janelia 托管的连接组图数据库与 Web 查询服务，支持按细胞类型/连接模式检索神经元与突触邻接，嵌入 Neuroglancer 可视化。
- **代码：** [neuprint-python](https://github.com/connectome-neuprint/neuprint-python)（**已开源**）；服务端为 Janelia 托管
- **沉淀到 wiki：** 是 → [`wiki/entities/neuprint.md`](../../wiki/entities/neuprint.md)

---

## 核心能力

- **连接查询：** 上游/下游邻接、路径、图模式匹配
- **注释检索：** 按 type、hemilineage、ROI 等过滤
- **数据集：** 托管多个连接组（含 `male-cns:v1.0`、FlyWire 相关导出等）
- **授权：** 需注册账号获取 API token（程序化访问）

---

## Python 快速开始

```bash
pip install neuprint-python
```

```python
from neuprint import Client, fetch_neurons, fetch_adjacencies
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
neurons, syndist = fetch_neurons("DNge104")
out_edges, info = fetch_adjacencies("DNge104")
```

亦可通过 `conda install -c flyem-forge neuprint-python` 安装。

---

## 生态关系

- **后端存储：** 常与 **DVID** 或 precomputed 体数据配合
- **可视化：** Web UI 嵌入 **Neuroglancer**
- **形态分析：** [navis](https://github.com/navis-org/navis) 提供 `navis.interfaces.neuprint` 封装

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| neuprint-python | **已开源**（PyPI / flyem-forge） |
| neuPrint 服务 | Janelia 托管 SaaS；需账号与 token |
| 服务端源码 | 非本 ingest 所列公开仓 |

---

## 对 wiki 的映射

- [neuPrint](../../wiki/entities/neuprint.md)
- [neuprint-python](../../sources/repos/neuprint-python.md)
- [Male CNS Connectome](../../wiki/entities/male-cns-connectome.md)
