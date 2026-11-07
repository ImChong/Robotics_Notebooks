# BEHAVIOR-Spatial Benchmark：面向机器人的空间理解评测｜论文原文归档

## 元数据

- **英文标题：** BEHAVIOR-Spatial Benchmark: Evaluating Spatial Understanding for Robotics
- **中文标题：** BEHAVIOR-Spatial 基准：评测机器人空间理解能力
- **作者：** Sirui Chen, Huang Huang, Qineng Wang, Wenlong Huang, Manling Li, Ruohan Zhang, Li Fei-Fei, Jiajun Wu
- **机构：** Stanford University（斯坦福大学）、Northwestern University（西北大学）
- **arXiv：** https://arxiv.org/abs/2610.34051
- **PDF：** https://arxiv.org/pdf/2610.34051
- **版本：** arXiv:2610.34051v2，2026-10-28（初稿 2026-10-25）
- **项目页：** https://arielchen07.github.io/behavior-spatial/
- **官方评测代码：** https://github.com/arielchen07/BEHAVIOR-Spatial-benchmark
- **官方数据：** https://huggingface.co/datasets/behavior-spatial-benchmark/BEHAVIOR-Spatial
- **核对日期：** 2026-11-07
- **核对范围：** 通读 v2 37 页：正文 §1–6（背景、数据构建、Qwen3.5-4B/π0.5 架构、评测协议、VQA/IR/Action 实验与局限），附录 A–F（问题模板/筛选、24 任务、数据与策略细节、打分规则、prompt、相关性/代码智能体/导航/规模消融）；对照官方仓库 README、评测脚本和 Hugging Face 数据卡。参考文献仅核对与本条目有关的原始引用，并非逐篇另行阅读。
- **通读：** 已完成

## 研究问题与核心贡献

机器人到达正确物体，未必说明它按照语言里的「左数第二个／距离最近／最大的」等空间关系选了目标。BEHAVIOR-Spatial 在同一类场景和空间关系上分离：
1. **Understand / VQA**：多选空间问答；
2. **Ground / IR**：从指令中预测目标点（reach 物体或 place 位置）；
3. **Act / Action**：让 VLA 在仿真中执行空间指令。

基于 BEHAVIOR-1K、OmniGibson 2025/2026 challenge 示范，构造 3,256 条空间 VQA（49 类活动）、1,240 条 grounding 查询（14 个操作任务），及 14 操作 + 10 导航任务。VQA 细分 1,572 一般题（单帧 1,078；多帧旋转 239 + 导航 255）和 1,684 任务题（操作 1,082 + 导航 602）。

## 方法、实验设计与关键发现

- **空间分类：** direction（相对方向、容器层位 locate、序数 ordinal），distance（最近/最远），size（物理尺寸比较）；单帧、多帧旋转/导航。
- **数据管线（论文图 3）：** 回放示范与仿真几何→提取 RGB/实例掩膜、3D 位置、相机参数→选取改变 referent 的指令与候选物→生成并验证 VQA、目标点、仿真状态。
- **基线 VLA：** π0.5 风格 flow-matching action expert，VLM backbone 改为 Qwen3.5-4B；比较 frozen VLM、LoRA、DROID 初始化和 dot/attention-map oracle IR。32-step action chunk、10 次 denoising、异步 10-step replanning（§5.3、附录 F.3）。
- **操作 held-out（469）与 prompt-probe（700）：** 前者更换未见配置与指令变体，后者固定训练布局但改变目标指令，专门测试「真按新指令转向目标吗」。
- **指标分清：** soft success 是完成正确交互或符合特定规则的未完成但朝正确目标推进；strict success 是完成实际目标；wrong-target 是选择了非指令目标（附录 D，Table S4）；导航只用目标到达二值成功。
- **主结果（论文 Table 4，操作 episode pooled，百分比）：**

| 策略 | Held-out soft | Held-out strict | Probe soft | Probe strict | Probe wrong-target |
|---|---:|---:|---:|---:|---:|
| frozen-VLM baseline | 41.6 | 27.9 | 16.1 | 10.0 | 70.0 |
| + LoRA | 43.5 | 30.9 | 17.9 | 9.6 | 68.1 |
| + DROID init | 40.1 | 24.7 | 16.6 | 8.9 | 66.0 |
| + oracle dot IR | 60.3 | 44.8 | 38.0 | 24.9 | 43.9 |
| + oracle attention map | 59.3 | 39.0 | 35.0 | 17.0 | 46.0 |

**注意：** oracle IR 使用真值目标点，并非 VLM 自动学会定位；实验还提供 Qwen3.5-4B 预测 IR，held-out soft 为 48.8%，仍低于真值 IR 60.3%（§5.3.4）。
- **VLM 评测：** Qwen3.5-9B 在操作 task-VQA 达 57.2%，IR strict 44.8%；Molmo2-8B 严格 IR 44.6%，宽容 IR 52.2%（Table 2）。通用 VQA 排名不能直接代理机器人成功率。
- **相关性：** 在 dot-IR 策略上 Qwen3.5-4B 的任务级 tolerant IR 与 soft success 的 Spearman ρ=0.701，高于 VQA 的 0.560；attention-map 为 0.753 vs 0.608。相关性取决于策略，非因果证明（Table 3）。
- **导航：** 10 任务成功率 22.5%–92.5%；多帧方向 VQA 与导航成功的相关性较弱（§5.4）。
- **Code-as-policy：** 单任务最多 10 次开发运行，冻结最终 Python 程序后评测；与 VLA 的闭环行为在同一 held-out 协议上比较（§4.4、附录 F.5），注意不能把该结果当作纯 VLA 推理分数。

## 局限与复现说明

- 全部仿真，**不代表已经验证真机安全或 Sim2Real**；运动、路线和时限也影响完成率（§6.1）。
- 操作指令目标点在 oracle IR 训练/评测中来自仿真状态，真实图像需要额外定位或标注。
- 项目代码仓库截至 2026-11-07 已公开 VQA/IR 评分器、OmniGibson rollout harness 和策略 websocket 接口；**不包含论文 VLA 策略的训练代码与论文 checkpoint**。更多见[源码归档](../repos/behavior-spatial-benchmark.md)。
- Hugging Face 数据公开但运行仿真依赖另行受许可约束的 BEHAVIOR-1K 3D 资产；数据 MIT、评测代码 Apache-2.0，不代表仿真资产可商用。

## 对 wiki 的映射

- [BEHAVIOR-Spatial 主实体](../../wiki/entities/paper-behavior-spatial.md)
- [官方项目页归档](../sites/behavior-spatial-benchmark.md)
- [官方评测源码归档](../repos/behavior-spatial-benchmark.md)
- [官方数据归档](../datasets/behavior-spatial-benchmark.md)
