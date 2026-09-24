# HDF Group — Introduction to HDF5（官方文档）

- **标题：** Introduction to HDF5
- **类型：** site（官方文档）
- **来源：** The HDF Group
- **链接：** https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html
- **用户指南索引：** https://docs.hdfgroup.org/hdf5/develop/index.html
- **入库日期：** 2026-09-24
- **一句话说明：** HDF5 **文件格式 + 逻辑数据模型 + C/h5py 等库**：以 **Group/Dataset** 组织异构数组与元数据；支持 **chunk + 压缩**、compound datatype、可扩展 dataspace；机器人栈常用作 **仿真/遥操作 episode 容器**（与 MP4 外挂或组内数组并存）。
- **沉淀到 wiki：** 是 → [`wiki/concepts/hdf5-file-format.md`](../../wiki/concepts/hdf5-file-format.md)

## 开源核查

| 项 | 状态 |
|----|------|
| 规范与文档 | **公开**（HDF Group） |
| 参考库 | **已开源** — [HDF5 仓库](https://github.com/HDFGroup/hdf5)（BSD 风格许可） |

## 核心摘录（官方 Introduction）

### 三要素

1. **File format** — 持久化 HDF5 数据的二进制格式  
2. **Data model** — 应用访问 HDF5 的逻辑组织（Abstract/Logical Data Model）  
3. **Software** — C 库、**h5py**（Python）、工具（**h5dump**、HDFView 等）

### 数据模型对象

- **Group**：类似目录，组织对象；每文件有 **root group** `/`  
- **Dataset**：存放 **raw data values** + 描述 metadata  
- **Datatype**：元素类型（预定义 H5T_* / 衍生 **compound** 表结构）  
- **Dataspace**：布局与维度；可 **fixed** 或 **unlimited（extendible）**；用于全量或 **subset I/O**  
- **Properties**：如 dataset **storage layout** — 默认 **contiguous**，可改 **chunked** 或 **chunked + compressed**  
- **Attributes**：附在对象上的小型 name/value 元数据（无 partial I/O / 压缩）

### 编程模型（官方）

典型顺序：**Open → Access → Close**；C API 前缀 `H5F`/`H5D`/`H5G`/`H5A` 等；Python **h5py** 以对象方法创建 dataset/group。

### 工具

- **h5dump** — DDL 形式查看文件结构（`-H` header、`-d` 指定 dataset）  
- **h5cc / h5c++** — 链接 HDF5 的编译脚本  
- **HDFView** — Java 浏览器

## 对 wiki 的映射

- [HDF5 文件格式（概念）](../../wiki/concepts/hdf5-file-format.md)
- [HDF5 vs MCAP vs LeRobot 对比](../../wiki/comparisons/hdf5-mcap-lerobot-data-formats.md)
- [Isaac GR00T](../../wiki/entities/isaac-gr00t.md) — 仿真 Teleop **HDF5 → LeRobot** 管线
