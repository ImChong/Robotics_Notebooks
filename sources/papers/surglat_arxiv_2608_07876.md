# SurgLAT: Surgical Latent Attention Tracking for Depth-Aware Robotic Laparoscope Control（arXiv:2608.07876）

> 来源归档（ingest）

- **标题：** SurgLAT: Surgical Latent Attention Tracking for Depth-Aware Robotic Laparoscope Control
- **缩写 / 框架：** **SurgLAT**
- **类型：** paper / surgical-robotics / attention / rcm
- **arXiv：** <https://arxiv.org/abs/2608.07876>
- **项目页：** <https://surglat-home-page.pages.dev/>（归档见 [`sources/sites/surglat-home-page.md`](../sites/surglat-home-page.md)）
- **作者：** Rulin Zhou、Qiujie Song、Yujie Ma、An Wang、Wanhao Liu、Guoheng Ma、Yidu Wang、Guankun Wang、Xingrong Diao、Jiankun Wang、Chaowei Zhu、Xianming Liu、Hongliang Ren
- **机构：** 香港中文大学（CUHK）；南方科技大学（SUSTech）；深圳大学；深圳市人民医院；深圳环区研究院
- **入库日期：** 2026-08-17
- **一句话说明：** 把术者关注区建成随时间演化的因果隐状态，解码热图 + 深度尺度，经虚拟轴 RCM 约束与零空间初始化驱动腹腔镜视野。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-17）：** 有 Abstract / Pipeline / SurgAtt 三数据集可视化 / 真机四场景。页头 **Code / Dataset / Videos** 按钮落在本站，**未列出独立 GitHub URL**。GitHub 搜索 SurgLAT 无对应官方实现仓。
- **结论：** **项目页已发布；独立训练/控制仓截至入库日未找到。** 源码运行时序图标 **不适用**。

## 摘录 1：感知

冻结 DINOv3 ViT-B/16；记忆引导高斯先验调制 token；state-conditioned spatial mixer（16 evidence token）；selective causal memory（4 latent token，短/长缓存 16/64）。解码注意力热图与操作区；深度支路在区内聚合相对深度，出轴向目标 \(D\)。

## 摘录 2：控制与数字

任务空间 \((x,y,D)\)；虚拟插入坐标 \(\lambda\) 建 RCM；7-DoF 冗余用零空间初始化放大旋转工作空间。SurgAtt-1.16M：

| | SZPH IoU / MCE / FPS | AutoLaparo IoU | Hamlyn IoU |
|--|---------------------:|---------------:|-----------:|
| SurgAtt-Tracker | 0.566 / 49.92 / 12.4 | 0.462 | 0.443 |
| **SurgLAT** | **0.604 / 41.24 / 34.5** | **0.527** | **0.479** |

真机：遮挡、快速运动、目标切换下在线跟踪 + 稳定视野。

**对 wiki 的映射：** [`wiki/entities/paper-surglat.md`](../../wiki/entities/paper-surglat.md)；交叉 [零空间控制](../../wiki/concepts/null-space-control.md)（若页存在则链，否则链 MPC）。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（仅项目页）
