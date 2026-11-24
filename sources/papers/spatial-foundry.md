# Spatial Foundry: A Multimodal Dataset for Spatial Intelligence（空间智能多模态数据集）

> 来源归档（paper）

- **论文：** <https://arxiv.org/abs/2611.17318v1>
- **PDF：** <https://arxiv.org/pdf/2611.17318v1>
- **HTML：** <https://arxiv.org/html/2611.17318v1>
- **项目页：** <https://ranolabs.com/Blog/spatial-foundry>
- **作者：** Ethan Lin、Yang Song
- **机构：** Ranolabs
- **发布日期：** 2026-11-12
- **收录日期：** 2026-11-24
- **核对范围：** arXiv v1 HTML 正文 §1–5（含 Fig. 1–5 的正文与图注、Table 1、式 1–11）；核对数据规格、生成流程、应用与证据边界，另查官方项目页数据访问入口。
- **通读：** 已完成

## 内容摘录（归纳）

### 1. 数据组织（§3）

5,280 是住宅场景配置数，同一户型可有多个家具布置变体；不是独立户型数。场景保留米制几何和对象标识，透视及全景 RGB、深度、点云、网格、3D 框、视频与逐帧相机记录共享空间参照。深度须区分轴向深度和径向距离。

对 wiki 的映射：[Spatial Foundry](../../wiki/entities/paper-spatial-foundry.md)。

### 2. 数据构建（§4）

住宅设计规则和功能模板约束 GPT-6 生成结构化平面图；人工检查通行、门洞和楼梯；Blender / Infinigen 装配后再审核。网格采样得到参考几何，虚拟扫描按球面角度分箱选近点，匹配可见扫描站的全景颜色并模拟距离相关误差；参考标注保持不变。相机与对象状态可独立控制。

对 wiki 的映射：[Spatial Foundry](../../wiki/entities/paper-spatial-foundry.md)。

### 3. 应用与证据（§5）

六类用途为平面图条件场景生成、几何重建、实例感知、相机条件视频、共享世界建模、状态条件编辑。Fig. 1 为模态覆盖比较，§5 为用途描述，未报告训练模型的量化基准或真机策略成功率。项目页样本需联系团队，许可及完整数据访问条件需另行确认。

对 wiki 的映射：[Spatial Foundry](../../wiki/entities/paper-spatial-foundry.md)。

## 关联来源

- [项目页归档](../sites/spatial-foundry.md)
