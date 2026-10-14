# Natura-1 — Nature Will 官方研究预览

> 来源归档（ingest）

- **标题：** The Will to Act — Natura-1: Scaling general dexterous manipulation through human experience
- **类型：** site / research preview
- **官方项目页：** <https://www.naturewill.com/blog/natura-1>
- **发布方：** Nature Will
- **核查日期：** 2026-10-14
- **代码 / 权重 / 数据：** 项目页未提供公开代码仓库、模型权重或数据集下载入口；公开内容为研究介绍、视频演示与项目方自报评测，不能据此视为可复现开源项目。
- **一句话说明：** Nature Will 将 Natura-1 描述为一个从大规模第一人称人类经验预训练、再用少量真实机器人数据适配到灵巧手的通用物理世界基础模型。
- **沉淀到 wiki：** 是 → [`wiki/entities/naturewill-natura-1.md`](../../wiki/entities/naturewill-natura-1.md)

## 官方披露摘要

- 作者称 Natura-1 是 **500 亿参数 MoE**，预训练语料超过 **100 万小时** egocentric 人类经验；经筛选后形成语言条件动作轨迹，使用单目/双目视频恢复稠密 3D 手姿，并按目标、子任务和步骤组织标注。
- 模型输入视觉和语言，目标端为 **20 自由度灵巧手**；官方称真实硬件观测到动作的闭环预算为 **100 ms**。训练和推理栈包含内部分布式训练引擎、Rust 编排层与融合 CUDA kernel。
- 展示平台包括双臂 ROKAE AR5 + Wuji 灵巧手，以及 ARX AC One 双臂夹爪平台。文章描述 12 个灵巧操作能力维度、长时序操作和未见指令泛化测试。
- 官方报告从约 4.4 万小时扩展到 100 万小时训练经验时，held-out 人类数据与机器人验证数据的动作预测误差均下降；这是作者报告的离线预测缩放结果，不等于机器人任务成功率同比增长。
- 官方比较缺少公认的同类灵巧手基础模型，部分基线的平台和末端执行器不同；横向分数应结合基准定义、任务覆盖与硬件差异解读。

## 硬件协作补充

- **本次项目补充信息：** 舞肌科技提供 Wuji Hand 2 灵巧操作硬件，支持 Nature Will 团队本次 Natura-1 训练与任务部署。Nature Will 页面此前展示的平台为 ROKAE AR5 双臂搭配 Wuji 灵巧手；公开说明未明确所用手部硬件批次、固件/SDK 版本，或具体训练和部署接口。
- **硬件规格参照：** [舞肌科技 Wuji Hand 2 官方产品手册](https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview/)将其描述为 20 个主动自由度的仿生灵巧手；当前文档标注为 2.2 Beta。该通用产品规格不能代替 Natura-1 展示硬件的具体版本确认。
- **边界：** 硬件协作不代表 Natura-1 模型、权重、训练数据或完整部署代码开放；也没有证据表明 Natura-1 使用了舞肌已公开的软件仓库。

## 开放性与可复现性

项目页提供模型说明、实验叙述和媒体演示，但没有链接到代码、权重或数据发布入口。训练语料、模型结构细节、训练配方与多数系统实现均为内部资源；现阶段应归类为**闭源研究预览**。如未来公开资源，需重新核对项目页并补充独立 `sources/repos/` 归档。

## 对 wiki 的映射

- [`wiki/entities/naturewill-natura-1.md`](../../wiki/entities/naturewill-natura-1.md)：项目定位、训练/适配流程、评测边界与开放状态。
- [`wiki/concepts/embodied-scaling-laws.md`](../../wiki/concepts/embodied-scaling-laws.md)：补充百万小时人类经验的闭源产业案例，并与 Dyna-2、EgoScale 等不同评测协议对照。
