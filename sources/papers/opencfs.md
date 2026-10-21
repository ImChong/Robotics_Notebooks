# openCFS 多物理场与数据处理论文

- **类型：** paper
- **来源：** SoftwareX / arXiv 作者原文
- **入库日期：** 2026-10-18
- **最后更新：** 2026-10-18
- **一句话说明：** 统一保存软件综述、声学模块及数据前后处理论文，导航到同一软件实体。
- **项目页：** https://opencfs.org/
- **代码：** https://gitlab.com/openCFS/cfs
- **站点归档：** [官方资料](../sites/opencfs.md)
- **仓库归档：** [官方实现](../repos/opencfs.md)

## 核心论文摘录

### 1) openCFS: An open-source finite-element software for advanced multiphysics simulations（Mayrhofer 等，2026）

- **链接：** https://doi.org/10.1016/j.softx.2026.103102
- **来源：** SoftwareX 36, 103102；题名、作者与出版信息由官网 Overview Publication 核实。
- **核心贡献：** 项目整体综述，覆盖软件、近期特性与历史背景；本次未取得该文全文，不据此推断性能指标。
- **对 wiki 的映射：** [openCFS](../../wiki/entities/opencfs.md)

### 2) openCFS: Open Source Finite Element Software for Coupled Field Simulation -- Part Acoustics（Stefan Schoder、Klaus Roppert，2022；v3 2025）

- **链接：** https://arxiv.org/abs/2207.04443
- **核心贡献：** 从声学模块介绍有限元 PDE 框架，涵盖吸收边界、非共形接口、瞬态/谐波/特征值求解等。
- **对 wiki 的映射：** [openCFS 的核心原理](../../wiki/entities/opencfs.md#核心原理)

### 3) openCFS-Data: Data Pre-Post-Processing Tool for openCFS（Stefan Schoder、Klaus Roppert，2023）

- **链接：** https://arxiv.org/abs/2302.03637
- **核心贡献：** 介绍 openCFS 的前后处理模块，重点是气动声学源项计算与过滤器。
- **对 wiki 的映射：** [openCFS 的工程实践](../../wiki/entities/opencfs.md#工程实践)

### 4) pyCFS-data: Data Processing Framework in Python for openCFS（2024）

- **链接：** https://arxiv.org/abs/2405.03437
- **核心贡献：** openCFS 配套 Python 数据处理框架的作者论文入口；本次只归档摘要级线索，未验证接口或运行表现。
- **对 wiki 的映射：** [openCFS](../../wiki/entities/opencfs.md)

## 当前提炼状态

已核对官方项目与论文摘要入口；统一映射至一个软件实体，没有拆分配套论文节点。未进行论文实验复现。
