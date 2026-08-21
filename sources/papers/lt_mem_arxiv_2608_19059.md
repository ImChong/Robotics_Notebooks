# LT-Mem（arXiv:2608.19059）

> 来源归档（ingest）

- **标题：** LT-Mem: Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding
- **类型：** paper / lifelong-scene-understanding / spatio-temporal-memory / slam / vqa
- **arXiv abs：** <https://arxiv.org/abs/2608.19059>
- **PDF：** <https://arxiv.org/pdf/2608.19059>
- **项目页：** <https://lt-mem.github.io/>（归档见 [`sources/sites/lt-mem-github-io.md`](../sites/lt-mem-github-io.md)）
- **数据集：** [LT-VQA Google Drive](https://drive.google.com/drive/folders/1rrwXxJDqJO9P9-wf_-JENX6FP1v9AThC)
- **机构：** DGIST（Robotics and Mechatronics Engineering）
- **作者：** Yumin Lee、Hyoseok Ju、Giseop Kim†
- **发表 / 上传：** 2026-08-21（arXiv v1）
- **入库日期：** 2026-08-21
- **索引来源：** [具身智能小站 8 篇综述](../blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)（<https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g>）

## 开源状态（步骤 2.5，2026-08-21）

- **部分开源：** 项目页 **Code (TBD)**；**LT-VQA** 数据集可通过 Google Drive 下载（3 env / 30 sessions / 80 QA）。
- **结论：** 数据可获取；**记忆系统代码未发布**。

## 摘录 1：问题

- 长期运行机器人反复访问变化环境：覆盖旧地图丢历史，逐次快照难维持跨会话身份 → 「时间性失忆」。

## 摘录 2：Tri-Memory 架构

- 多会话 **MASt3R-SLAM** + 实例 3D 分割对齐对象级观测。
- **Live / Delta / Meta** 三层：当前状态、变化事件、元信息；按对象 **波动性** 选择 **覆盖 / 保持 / 多假设**。
- 确定性证据评分维持身份一致；波动性条件时序推理。

## 摘录 3：LT-VQA 与效率

- 多会话记录 + 持久身份标注 + 时间问答。
- 相对 VLM-Batch 基线全面更优，且令牌消耗 **低约一个数量级**（~16×）。

**对 wiki 的映射：** [`wiki/entities/paper-lt-mem.md`](../../wiki/entities/paper-lt-mem.md)；交叉 [Spatial Memory Agent](../../wiki/entities/paper-spatial-memory-agent.md)、[VLN](../../wiki/tasks/vision-language-navigation.md)。

## 当前提炼状态

- [x] 项目页开源核查（数据集 yes / code TBD）
- [x] 升格 `wiki/entities/paper-lt-mem.md`
