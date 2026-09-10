# neuroglancer

> 来源归档

- **标题：** Neuroglancer
- **类型：** repo
- **机构：** Google（非官方 Google 产品）
- **链接：** https://github.com/google/neuroglancer
- **文档：** https://neuroglancer-docs.web.app
- **Demo：** https://neuroglancer-demo.appspot.com
- **Stars：** ~1.4k+（2026-09）
- **许可：** Apache-2.0
- **入库日期：** 2026-09-10
- **一句话说明：** WebGL 体数据浏览器，支持任意截面、3D mesh 与 skeleton，是连接组 EM/分割数据的标准交互前端。
- **代码：** https://github.com/google/neuroglancer（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/neuroglancer.md`](../../wiki/entities/neuroglancer.md)

---

## 核心定位

- **纯客户端：** 通过 HTTP 读取远程体数据，无强制后端
- **四窗格 UI：** 3 个正交截面 + 1 个可独立朝向的 3D 视图
- **Male CNS 示例：** [预加载场景](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json)

---

## 支持的数据源（README）

- Neuroglancer **precomputed**（FlyEM/MaleCNS 常用）
- N5、Zarr v2/v3、OME-Zarr
- **DVID**、BOSS、Render、NIfTI、Deep Zoom

---

## 安装

```bash
pip install neuroglancer   # Python 绑定
npm install neuroglancer   # JS 包
```

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| GitHub | **已开源** Apache-2.0 |
| 可运行 | demo.appspot.com 或本地 `python -m neuroglancer` |
| 依赖 | 需 WebGL 2.0 + `EXT_color_buffer_float` |

---

## 对 wiki 的映射

- [Neuroglancer](../../wiki/entities/neuroglancer.md)
- [DVID](./dvid.md)
- [果蝇连接组工具栈](../../wiki/concepts/fly-connectomics-stack.md)
