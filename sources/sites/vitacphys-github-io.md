# ViTacPhys Project Page

> 来源归档

- **标题：** ViTacPhys: Physical Property-Aware Grasping from Human Visual-Tactile Demonstrations
- **类型：** site / project page
- **URL：** <https://vitacphys.github.io/ViTacPhys/>
- **论文：** <https://arxiv.org/abs/2608.21355>
- **机构：** 小米机器人实验室（Xiaomi Robotics）
- **通讯作者：** Shuaijun Wang
- **入库日期：** 2026-08-24
- **一句话说明：** 官方项目页：人体视触觉采集与标注、ViTacPhys 预测器、人→机迁移与 ACT 式自适应抓取真机对比；含数据集规模、预测/抓取表与失败案例。

## 开源状态（项目页核查，2026-08-24）

| 项 | 状态 |
|----|------|
| arXiv PDF/HTML | **已发布**（2608.21355） |
| Code 按钮 | **Coming Soon** — 无 GitHub 实现链 |
| Dataset 按钮 | **Coming Soon** — 无公开下载 |
| 结论 | **宣称将开源 / 待发布** |

## 页面结构（策展）

| 区块 | 内容要点 |
|------|----------|
| Abstract | 60 物体 / 1800 人体示范；质量·刚度·摩擦预测；人→机迁移 + 在线条件化抓取 |
| Method | 时序视触觉 + VLM 语义先验 → 物理属性 → ACT 策略 |
| Dataset | 可穿戴采集（腕 RGB + 指尖压感 + 动捕）；垂直提起 / 侧向摇晃两协议 |
| Results | 属性预测 ID / held-out / one-shot；人→机微调消融；真机 ID/OOD 抓取 vs ACT / ViTacFormer |
| Platform | 7-DoF 臂 + 6-DoF 灵巧手；Quantum Manus 遥操作；Jetson Orin 30 Hz 部署 |

## 对 wiki 的映射

- 论文：[`sources/papers/vitacphys_arxiv_2608_21355.md`](../papers/vitacphys_arxiv_2608_21355.md)
- 沉淀 **[`wiki/entities/paper-vitacphys.md`](../../wiki/entities/paper-vitacphys.md)**
