# SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery（arXiv:2609.31507）

> 来源归档（ingest）

- **标题：** SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery
- **类型：** paper / aerial VLN / benchmark / satellite imagery / long-horizon navigation
- **作者：** Jiajun Jiang、Chunliang Hua、Zichun Chen、Yanxing Wu、Zeyuan Yang、Jie Song、Xiao Hu
- **机构：** 香港科技大学广州校区（HKUST(GZ)）、LASER / 国际数字经济学院（IDEA）、香港科技大学（HKUST）
- **发表：** NeurIPS 2026 Evaluations & Datasets（已接收，待正式出版）
- **arXiv：** <https://arxiv.org/abs/2609.31507>（v1，2026-09-25）
- **项目页：** <https://eku127.github.io/SatNav/>
- **代码：** <https://github.com/Eku127/SatNav>（MIT；含环境、基线、训练/评测文档与可运行示例）
- **数据：** [SatNav-Episodes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1)；SatNav-Scenes-v0.1 高分辨率 GeoTIFF 场景需申请
- **模型：** [SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)
- **入库日期：** 2026-10-14
- **一句话说明：** 用高分辨率卫星图构建城市尺度 UAV 视觉–语言导航基准，以长程地理空间接地、记忆和闭环执行为核心，并报告卫星训练策略向真实航拍观测迁移。

## 核心摘录（面向 wiki 编译）

1. **规模化场景与轨迹：** 从 18 座城市的 59 个场景构建 118,494 条 episode，平均轨迹长度 379 m；语言指令、起点、目标、航点和参考路径成套提供，含 train、val_seen 与 val_unseen 划分。
2. **三类长程任务：** Boundary 要求沿边界巡航并闭环返回；Landmark 以地标进行空间接地；Route 按含计数线索的语言指令跟随路线。三者分别考查环路进度记忆、地标定位和路线次序理解。
3. **卫星图像作为空中观测近似：** SatSim 从局部 GeoTIFF 场景生成 RGB 观测，模拟 UAV 向下看的视野，使场景生产不再依赖逐城重建完整 3D 环境。
4. **SwiftVLN 记忆组件：** 用滑动对话窗口串接局部上下文，并对历史观测做可切换抽样、空间线索增强及记忆压缩；项目页列出均匀/随机/近期偏重抽样、初始观测/相对位姿提示、逐帧池化/地图记忆/分段或全局 token 聚类等消融选项。
5. **评测与迁移：** 比较经典 VLN 与 LVLM 智能体；项目页指出测试集泛化仍具挑战，最强 StreamVLN* 在 Test Seen / Test Unseen 的 SR 为 67.3% / 58.4%。卫星图训练的 SwiftVLN 通过轻量视觉适配器迁到真实 UAV 观测，并展示 Boundary、Landmark、Route 飞行视频。
6. **开放边界：** 官方 SatNav 代码仓库采用 MIT；episode 数据、模型检查点有公开 Hugging Face 入口。高分辨率卫星 GeoTIFF 场景需申请访问或使用自有 API 凭据生成；数据、地图内容和第三方组件的条款须分别核对仓库的 DATA_LICENSE.md 与 NOTICE。

**对 wiki 的映射：** [SatNav 项目实体](../../wiki/entities/paper-satnav.md)；补充 [视觉–语言导航任务页](../../wiki/tasks/vision-language-navigation.md)。

## 来源交叉索引

- 项目页：[satnav-eku127-github-io.md](../sites/satnav-eku127-github-io.md)
- 官方代码：[eku127_satnav.md](../repos/eku127_satnav.md)

## 版本与引用

- arXiv v1 提交于 2026-09-25；论文页面标注已接收 NeurIPS 2026 Evaluations & Datasets。
- 项目页与 README 为代码、数据、模型及实机展示状态的核查依据；benchmark 数字应按对应 split 与指标阅读。
