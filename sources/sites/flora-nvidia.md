# FLORA：NVIDIA GEAR 官方项目页核验

> 来源归档（site / project-page），核验日期：2026-11-10；以官方网站实际公开入口与官方 GitHub README 为准。

- **标题：** Flora: An Open Model Suite for Scaling Robotics Data Engines
- **机构：** NVIDIA GEAR（英伟达）
- **官方项目页：** <https://research.nvidia.com/labs/gear/flora/>
- **论文（v1）：** <https://arxiv.org/html/2610.31779v1>
- **论文 PDF：** <https://arxiv.org/pdf/2610.31779v1>
- **代码：** <https://github.com/NVlabs/Flora>（项目页 Code 链接，NVIDIA/Flora 重定向至 NVlabs/Flora）
- **模型权重：** <https://huggingface.co/collections/nvidia/flora>
- **开源类型：** **部分开源**：官方仓公开 captioner/、tempseg/、rewarder/ 的**推理示例与权重**，但明确**不包含训练代码、生产推理系统、内部数据**。
- **核对范围：** 项目主页摘要、Paper / Code / Models 三个官方入口、指标展示和成本/生产规模介绍，以及官方 GitHub README 的许可与生产样例范围。
- **通读：** 已完成

## 官方项目要点与口径

1. **动作时间戳指令：** FLORA-TempSeg 输出动作时间区间，FLORA-Captioner 为区间生成动作意图描述；视频可来自人类操作或机器人多视角操作。
2. **奖励接口：** FLORA-Rewarder 可生成结果分数、每 0.5 秒的过程执行质量评分与测试时自定义 rubric 步骤分。
3. **训练用途：** NVIDIA 论文称该数据引擎已用于 GR00T 模型训练；发布的软件不是 GR00T 控制策略。
4. **公开基准：** WGO-Bench 分段 F1@0.75 50.3；RoboReward MAE×100 40.0；Robometer OOD Kendall×100 72.0。不同指标不能简单合并为一个综合分。
5. **成本和规模：** 首页写约 125× 相比高 effort Astra 的标注成本，以及 512 张 GB300 每天约 200,000 视频小时；论文 Appendix D 表明成本为包含特定 GPU 单价和速率假设的**估算值**，生产规模属于内部实测披露，**均不能当成 GitHub 示例脚本实测吞吐**。
6. **更新原历史误判：** 2026-10-29 之前的本库摘录曾认为项目页无可核实 GitHub 入口；截至本次核验，网站顶端有可直接跳转的 Code 链接，官方仓确实开放。

## 注意许可及独立复现边界

- 代码与 Hugging Face 权重使用 NVIDIA 非商业研究/评估许可；不能默认商用。
- 官方 README 明确 `This repository holds the inference code ... there is no training code here`。生产数据引擎的分布式服务与数据没有随此仓公开。
- Dense Reward 的后见标签用于离线整理，不可作为唯一实时安全监控/直接控制机器人依据。

## 对 wiki 的映射

- [FLORA 主实体](../../wiki/entities/flora-robotics-data-engine.md)
- [论文全文核验](../papers/flora_nvidia_2026.md)
- [NVlabs/Flora 源码核验](../repos/flora-nvlabs.md)
