# Quantitative control and recording of materials-synthesis processes using an automated experimentation platform

> 来源归档（ingest）

- **标题：** Quantitative control and recording of materials-synthesis processes using an automated experimentation platform
- **简称：** SAIGEN
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.14928>
- **PDF：** <https://arxiv.org/pdf/2609.14928>
- **代码：** <https://github.com/YusukeHashimotoLab/saigen>

- **入库日期：** 2026-09-15
- **索引来源：** [具身智能小站 9+EffVLA 盘点](../blogs/wechat_embodied_station_9_papers_resources_effvla_2026-09-15.md)
- **一句话说明：** 商用设备 + 3D 打印夹具 + AI 生成控制流程；示范 ZIF-8 合成并记录加液速度等过程变量。

## 开源状态（步骤 2.5，2026-09-15）

**结论：已开源**

## 核心摘录

### 摘录 1

商用设备 + 3D 打印夹具 + AI 生成控制流程；示范 ZIF-8 合成并记录加液速度等过程变量。

**对 wiki 的映射：** [paper-saigen](../../wiki/entities/paper-saigen.md)

### 摘录 2（官方 abstract 要点，2026-09-15 补录）

- **论文题名：** *Quantitative control and recording of materials-synthesis processes using an automated experimentation platform*。
- **动机：** 数据驱动的材料开发需要大量 **高质量材料数据**；**完全自主** 的材料实验虽被期待，但技术门槛高、落地有限。
- **定位（关键）：** 本平台 **刻意不追求完全自主**，而聚焦 **可靠的过程自动化** 与 **过程量的定量记录**。
- **硬件构成：** 商用 **机械臂、电动移液器、网络摄像头、电子天平** 组合；**夹具等零件用 3D 打印** 自制。
- **控制方式：** 仪器由 **基于 LLM 的 AI agent 生成的控制代码** 驱动。
- **演示体系：** 双溶液混合体系，合成 **ZIF-8**（一种金属有机框架 MOF）。
- **结果：**
  - 产物中观察到 **白色悬浮相**，**X 射线衍射（XRD）** 确认为 ZIF-8；
  - **粒径分布强烈依赖电动移液器的加液速度** —— 这是 **人工操作下难以控制与记录** 的参数；
  - 该依赖关系在 **重复实验中可复现**，验证了自动合成的 **重复性**。
- **立论：** 人工操作中很少被量化的 **过程参数**，可能直接决定材料数据的质量。
- **开源：** **全部控制代码、CAD 模型与文档公开**，以推动实验室尺度的实验自动化普及。

**对 wiki 的映射：** 同上（补入该页「核心原理（方法）」「实验与评测」「与其他工作对比」三节）

## 当前提炼状态

- [x] 项目页/仓库核查
- [x] wiki 映射
