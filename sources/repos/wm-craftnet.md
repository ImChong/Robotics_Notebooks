# sharpa-robotics/WM-Craftnet

- **标题：** WM-Craftnet（官方代码仓，待发布）
- **类型：** repo / dexterous-manipulation / world-model / in-hand-rotation
- **URL：** <https://github.com/sharpa-robotics/WM-Craftnet>
- **机构：** Sharpa Robotics
- **配套论文：** [wm_craftnet_arxiv_2609_07002.md](../papers/wm_craftnet_arxiv_2609_07002.md)
- **项目页：** <https://wmcraftnet.github.io/>
- **入库日期：** 2026-09-11
- **复核日期：** 2026-09-11
- **开源状态：** **待发布** — 项目页（2026-09-11）写明 *We will release the codes soon!*；GitHub REST API 与匿名访问均为 **404 Not Found**，截至入库日 **无可运行实现**。
- **一句话说明：** WM-Craftnet 预期官方实现仓（WSM 预训练 + PPO 手内旋转 + Sharpa 真机部署）；入库时仅作 URL 占位与 lint 跟进锚点。

## 核查记录（步骤 2.5）

| 核查项 | 结论 |
|--------|------|
| 项目页 Code 区 | 文案 *We will release the codes soon!*，无有效链 |
| `GET /repos/sharpa-robotics/WM-Craftnet` | **404**（2026-09-11） |
| README / 训练脚本 | 不可访问 |
| 权重 / 数据集 | 未列出 |

## 预期复现路径（待开源后核实）

1. WSM 预训练：noisy depth + proprio + tactile + action → 重建 clean depth / proprio-tactile / reward  
2. 冻结或共享 WSM，训练 asymmetric PPO actor–critic，读取 \(h_t\)  
3. 仿真九物体 z 轴 → 可选四十九物体下游微调  
4. Sharpa 真机：腕部 depth + 触觉闭环部署  

## 对 wiki 的映射

- [paper-wm-craftnet.md](../../wiki/entities/paper-wm-craftnet.md)
- [wmcraftnet-github-io.md](../sites/wmcraftnet-github-io.md)

## 参考来源

- 项目页：<https://wmcraftnet.github.io/>
- 预期仓库：<https://github.com/sharpa-robotics/WM-Craftnet>（截至 2026-09-11 不可用）
