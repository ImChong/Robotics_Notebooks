# human five：Zhengyi Luo 从通用运动控制到自主操作（2026-11-22）

- **类型：** 二手文章 / 技术演讲中文长文整理
- **文章标题：** Zhengyi Luo：从通用人形机器人运动控制到自主操作：Sonic、GR00T 与 WAM
- **作者：** human five
- **发布日期：** 2026-11-22 09:54（用户提供）
- **原文 URL：** <https://mp.weixin.qq.com/s/tnafAt6AmFgInGRU1IsFXQ>
- **对应演讲：** [Penn GRASP Seminar，2026-11-18](../talks/zhengyi-luo-motion-tracking-world-action-models-2026-11-18.md)
- **入库日期：** 2026-11-22
- **核对范围：** 已阅读用户在对话中提供的全文（包含 Q&A），并核实演讲题目、时间和相关 SONIC、GR00T 2、GRAIL 官方文献；尚不能直接核对演讲录像和逐字转写。**文章不是一手研究论文，也不是讲者校订的逐字稿。**
- **通读：** 已完成

## 归纳摘要（非全文转载）

- 运动追踪为人形控制建立低层动态稳定性先验，但并不回答「该做什么」。
- SONIC 的统一 motion token 使遥操作、运动生成、语义策略可复用同一执行层。
- GRAIL 将生成视频与有尺度的 3D 场景、4D 重建、重定向、物理修正联系起来，降低全身操作数据采集成本。
- GR00T 2 是联合未来视频与动作预测的 World-Action Model；Turbo 版本侧重机器人本体部署。
- 双层架构最关键的是 **state-to-action latency、执行频率与动作分块时长分别度量**。
- 问答涵盖 token 表示、示范清洗、跨本体视频、感知是否下沉至 System 0、追踪反馈反向优化 retargeting、以及残余误差由高层补偿的取舍。

## 重要勘误 / 保留意见

- 转述将 SONIC 消费 token 写为每 5 ms 一次，但官方资料标注 **50 Hz（每 20 ms）**；另有 **1000 Hz 关节 PD 回路**，应分清。
- GR00T 2 Turbo 的压缩和模型规模应以官方白皮书具体版本为准；其精度不保证与完整版逐项等价。
- 所有任务成功率须注明测试次数、对象、环境与是否 OOD；详见 [白皮书数据口径](../papers/gr00t2_whitepaper_2026.md)。

## 对 wiki 的映射

- [SONIC × GR00T 2 Turbo 双层架构](../../wiki/queries/sonic-gr00t2-wam-humanoid-dual-brain.md)
