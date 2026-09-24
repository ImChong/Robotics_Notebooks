# RoboTwin-Phys: 物理参数多样性评测基准（arXiv:2609.26292）

> 来源归档（ingest）

- **标题：** RoboTwin-Phys: Do WAMs and VLAs Understand the Physical World?
- **类型：** paper / benchmark / manipulation / physics-diversity
- **arXiv abs：** <https://arxiv.org/abs/2609.26292>
- **PDF：** <https://arxiv.org/pdf/2609.26292>
- **项目页：** 无独立项目页（Memo 技术报告格式）
- **代码/数据：** 宣称发布 **5000+** 专家示范 + 13 维物理 GT；截至 2026-09-24 **无** 官方 benchmark 仓库链接（GitHub 检索仅见非官方 `yefeng00/RoboTwinPhysTest`）
- **机构：** 北京大学（PKU）、北大先进信息技术研究院、Memo 等
- **入库日期：** 2026-09-24
- **一句话说明：** 在 RoboTwin 2.0 五十任务上 episode 级连续采样 **13** 个物理属性；WAM/VLA 在视觉/布局随机化下仍有效，但物理条件变化时 SR 显著下滑（例：π₀.₅ 平均 31.60%）。

## 核心摘录

### 1) 设计

- **13 物理属性** episode 级连续采样（质量、摩擦、CoM、关节阻尼等），范围 task-specific 校准。
- **5000+** expert demos，兼容官方 RoboTwin 格式 + **13-d 物理参数标注**。
- 评测时先 expert planning 验证物理可实现性。

### 2) 评测读法（Table 摘录，Randomized 通道 avg SR %）

| 模型 | Avg SR |
|------|--------|
| Fast-WAM | 44.24 |
| Motus | 39.60 |
| FACT | 39.14 |
| π₀.₅ | 31.60 |
| Galaxea-VLA | 37.82 |

- 相对仅视觉/布局 DR 的 benchmark，**物理多样性暴露独立 robustness gap**。

### 3) 开源状态（2026-09-24）

- 论文承诺 benchmark + 数据；**官方下载链待跟进** → **部分 / 待发布**。

## 对 wiki 的映射

- 新建：[paper-robotwin-phys](../../wiki/entities/paper-robotwin-phys.md)
- 交叉：[robotwin](../../wiki/entities/robotwin.md)、[vla](../../wiki/methods/vla.md)、[world-action-models](../../wiki/concepts/world-action-models.md)
