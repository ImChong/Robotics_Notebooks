# VLN-AR｜A Hierarchical Onboard Vision-Language-Navigation Framework for Aerial Robots in Complex Environments

> 论文一手资料归档（paper）；不转载全文，保留研究方法、评价口径与复现边界。

- **论文标题：** A Hierarchical Onboard Vision-Language-Navigation Framework for Aerial Robots in Complex Environments
- **中文标题：** VLN-AR：面向复杂环境空中机器人的分层机载视觉–语言导航框架
- **作者：** Yuze Wu、Mo Zhu、Xingxing Li、Yuheng Du、Zhaoqi Wang、Yuxin Fan、Wenjun Li、Kenghou Hoi、Zhichao Han、Xin Zhou、Fei Gao
- **机构：** 浙江大学工业控制技术全国重点实验室；Differential Robotics
- **原文：** <https://arxiv.org/pdf/2611.23930v1>
- **HTML：** <https://arxiv.org/html/2611.23930v1>
- **arXiv：** 2611.23930v1，2026-11-18
- **期刊：** IEEE Transactions on Robotics，论文首页写明 2026-11-09 接收；DOI [10.1109/TRO.2026.3754505](https://doi.org/10.1109/TRO.2026.3754505)
- **项目：** <https://wyz000.github.io/VLN-AR/>
- **入库及资源核查日期：** 2026-11-22
- **核对范围：** arXiv v1 全文 I 引言、II 相关工作、III 模型与 3DGS 自动数据构造（式 1–5）、IV 三阶段 SFT/RFT、Agent Algorithm 1、机载部署、V 实验（Table I–VI、Fig. 1–14、消融及真实世界局限）、VI 结论及参考文献；并对照官方项目页 Overview、Results、Method、Citation、资源链接。
- **通读：** 已完成

## 原文机制摘录与核验

1. **分层解耦（III-A，式 1–5）：** `π_VLN-AR = φ_LF ∘ π_VLM`。VLM 接受指令及多帧图像，预测机体系 3D pose/yaw、图像平面的 2D bbox、CoT 推理、finish、replan；learning-free action expert 基于 Ego-Planner 改进，结合深度/几何避障输出动态可行轨迹。不能把 200 Hz 规划控制误称为 200 Hz 多模态推理。
2. **3DGS 数据（III-B）：** 实拍视频→3DGS 重建→Unity 及语义标注→可行起终点/任务语言生成→受动力学约束的专家轨迹→10 Hz 记录 RGB、深度、LiDAR、6-DoF 里程计，辅以语言重述、端点误差小于 1 m 的筛选和语义核验。文章整体报告约 **400 万样本**，但不等于作者已公开可下载数据。
3. **三阶段训练（IV-A）：** Stage I 为 VQA / Grounding / Reasoning 的全参数 SFT；Stage II 为 3D 位姿、finish、replan 的导航 SFT；Stage III 是集成 action expert 的闭环 GRPO RFT，包含成功/失败稀疏回报、分组相对优势与分阶段折扣奖励，丢弃零方差轨迹组。
4. **长程 Agent（IV-B，Algorithm 1）：** 任务拆分为子任务队列，存储历史关键帧、轨迹、状态，`finish` 触发推进，`replan` 触发主动观察/旋转搜索，必要时用 CoT 提议探索路点。论文写明 VLM/Agent **2 Hz**、action expert **>200 Hz**，异步执行。
5. **推理优化（III-A2、IV-C）：** 指令相似性引导的 ViT 12/15/22 层逐级保留 75%/50%/25% 视觉 token（训练时关闭）；配合算子调度、Flash-Attention、KV cache 与低精度量化。Table III 在 Jetson Orin NX 30 W 下 2B/4B/7B **每 token 解码**加速 7.1–9.4 倍；Fig. 14 另报整次模型推理从约 4100 ms 到 494 ms（8.3 倍）。不同计量口径勿混写。
6. **仿真（V-A/B，Table II）：** AerialVLN SR **78.15%** / NE 22.12 m，OpenFly SR **76.88%** / NE 23.22 m，自建未见室内测试 SR **92.66%** / NE 5.75 m。外部两基准未用于训练；成功判定距离分别为 5 m（室外）及 1 m（自建室内），指标不是可直接横比的统一难度。
7. **真机（V-C）：** 两款 Differential Robotics UAV 搭载 Intel RealSense 和 Jetson Orin NX；101 次飞行成功 91 次（**90.1%**）。Table IV 每类 20 次：悬浮障碍 18、>5 m 高差 19、夜间弱光 17、动态人类障碍 17、未知场景零样本 18。这里是研究实验样本，不是通用安全认证。
8. **消融（V-D，Table V/VI）：** 三阶段完整训练的精确导航 **85.7%**、长程导航 **88.0%**；Agent 去除时长程 **57.4%**，加入后 **88.0%**。在 976 次模拟长程 episode 中发现 13 次错误 finish 判断（11 次过早完成、2 次未完成标记）；10 次真机长程试验未见此类错误，但样本数不足。
9. **作者明确局限（V-C4/VI）：** 相似物体造成语义错指、远距离/小目标的 2D→3D 投影与定位误差；未来计划更多传感模态、因果时间推理及 >10 Hz 端到端推理。不能将高频 trajectory executor 等同端到端感知推理能力。

## 开源与可复现边界

- 2026-11-22 官方项目页含论文、视频、方法和结果，**未列训练/部署源码、公开权重或数据集下载入口**。论文表示室内测试集将来开放，当前不把“计划公开”当“已公开”。
- 无可核对的官方 `train.py` / `eval.py` / 推理入口，因此**源码运行时序图不适用**。论文框架图是论文描述的架构，不是已复现代码调用路径。

## 对 wiki 的映射

- [VLN-AR 独立论文实体](../../wiki/entities/paper-vln-ar-aerial-navigation.md)
- [VLN-AR 官方项目页归档](../sites/vln-ar-aerial-navigation.md)
- [视觉–语言导航任务](../../wiki/tasks/vision-language-navigation.md)
