# DVID

> 来源归档

- **标题：** DVID（Distributed, Versioned, Image-oriented Dataservice）
- **类型：** repo
- **机构：** HHMI Janelia FlyEM
- **链接：** https://github.com/janelia-flyem/dvid
- **文档：** [GUIDE.md](https://github.com/janelia-flyem/dvid/blob/master/GUIDE.md)
- **Stars：** ~202+（2026-09）
- **入库日期：** 2026-09-10
- **一句话说明：** Janelia FlyEM 为神经重建打造的分布式、分支版本化图像与注释数据服务，支撑 TB 级体数据、labelmap、突触与 JSON 元数据统一版本管理。
- **代码：** https://github.com/janelia-flyem/dvid（**已开源**，Go）
- **沉淀到 wiki：** 是 → [`wiki/entities/dvid.md`](../../wiki/entities/dvid.md)

---

## 核心定位

- **规模：** 数十亿离散数据单元、TB–PB 级成像体数据
- **版本模型：** 分支版本化（类似 git），**所有版本可同时查询**（无 checkout）
- **可插拔 datatype：** `labelmap`、`annotation`、`keyvalue` 等，按科学 API 而非裸文件操作
- **存储后端：** Badger、filestore、ngprecomputed、云存储等可混配

---

## 与 Neuroglancer 关系

- Neuroglancer **原生支持 DVID** 作为数据源
- DVID 的 `ngprecomputed` 引擎可暴露 precomputed 格式供 Web 客户端读取

---

## 构建要点（README）

```
Go 1.25+, CGO, C compiler
git clone https://github.com/janelia-flyem/dvid && cd dvid && make
bin/dvid about
```

生产部署建议用 [releases](https://github.com/janelia-flyem/dvid/releases)；`main` 分支可能有破坏性变更。

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 服务端 | **已开源**（Go，自托管） |
| 客户端 | Neuroglancer、内部 FlyEM 工具链 |
| 局限 | 无域专用 diff/PR 协作 UI；大科学数据「github」愿景仍在演进 |

---

## 对 wiki 的映射

- [DVID](../../wiki/entities/dvid.md)
- [Neuroglancer](./neuroglancer.md)
- [果蝇连接组工具栈](../../wiki/concepts/fly-connectomics-stack.md)
