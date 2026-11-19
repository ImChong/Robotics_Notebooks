# CloudRobo-WM1.2｜华为云昇腾原生世界动作模型发布

- **类型：** 华为云官方发布资料（同一 CloudRobo-WM 家族的 WM1.2 版本）
- **官方新闻全文：** https://www.huaweicloud.com/news/2026/20261116171910882.html
- **官方发布会及议程：** https://www.huaweicloud.com/about/live/261116.html
- **官方产品文档：** https://support.huaweicloud.com/cloudrobo_faq/cloudrobo_06_0005.html
- **发布时间：** 2026-11-16
- **核对范围：** 已通读华为云 2026-11-16 官方新闻正文、发布会议程和官方 CloudRobo 预置模型说明。未取得 WM1.2 独立版本的训练/推理源码、正式模型卡、权重清单或完整技术论文；发布会回放需登录，未观看。
- **通读：** 已完成
- **入库日期：** 2026-11-19

## 官方发布确认的事实

- CloudRobo-WM1.2 宣布为昇腾原生具身世界模型，使用共享世界模型骨干，提供预测增强的世界表征与原生动作生成。
- PPA 多模态世界表征：强调交互结果与未来物理状态预测，不仅是静态目标检测。
- DualAct：世界模型动作路径与基于 MoT 的动作专家路径；官方称后训练可灵活选择动作专家大小。
- 多粒度监督：根据缺失动作标注、深度、部分维度及相机视角调整监督，从异构数据学习。
- 本体语义保真：跨单臂、双臂、轮式等机器人的动作接口与运动建模。
- 可组合的 World/Expert Action 与 Joint/Action-only 推理方式；官方未在新闻正文提供逐档延迟、吞吐与硬件耗时表。

## 评测与开放边界

华为云官方声明其在 RoboTwin 2.0 和 LiberoPro 2.0 排名第一，但公告未给出充分的复现实验表格、版本 checkpoint 对应或随机种子。本次仅记录为**官方声明**，不把 XPolicyLab 的 CloudRobo-WM 评测数值、HF 上的 WM-1.0 权重强行归属于 WM1.2。

发布会以“发布暨开源”为议程，但截至 2026-11-19 未从华为官方页面找到直接标明 WM1.2 的独立代码提交、权重下载或许可证。应将“宣布开源”和“可取得 WM1.2 完整资产”分开记录。

## 版本归属

- [CloudRobo-WM 原始项目说明](../repos/cloudrobo_wm_repository.md)
- [CloudRobo-WM XPolicyLab 推理/后训练适配代码](../repos/cloudrobo_wm_xpolicylab.md)
- [WM-1.0 RoboTwin 模型权重归档](../models/cloudrobo-wm-1-0-robotwin-20261111.md)
- [Wiki 家族实体与版本对照](../../wiki/entities/cloudrobo-wm.md)

## 未完成的独立核验

1. WM1.2 专属模型卡、版本标签、代码许可与权重 SHA256。
2. PPA / DualAct 在可执行代码中的模块边界、训练 loss 与实时推理切换开销。
3. RoboTwin 2.0 与 LIBERO-Pro 2.0 原始协议、checkpoint 与可复现结果。
4. 独立技术报告全文、发布会录播（当前未通读，不能写为已核验）。
