# Uranus: Building the Next-Generation Simulation Infrastructure for Embodied AI（arXiv:2609.24815）

> 来源归档（ingest）

- **标题：** Uranus: Building the Next-Generation Simulation Infrastructure for Embodied AI
- **简称：** Uranus
- **类型：** paper / learned-simulator / action-conditioned-world-model / embodied-ai
- **arXiv：** <https://arxiv.org/abs/2609.24815>
- **PDF：** <https://arxiv.org/pdf/2609.24815>
- **项目页：** <https://d-robotics-ai-lab.github.io/large-model-team/blog/uranus/> — 归档见 [项目页记录](../sites/d-robotics-uranus.md)
- **官方推理代码：** <https://github.com/D-Robotics-AI-Lab/Uranus-OSS> — 归档见 [代码记录](../repos/uranus-oss.md)
- **机构：** 地瓜机器人（D-Robotics）大模型团队
- **版本日期：** arXiv v1 于 2026-09-21 提交；v3 于 2026-09-23 修订。项目博客页面显示日期为 2026-08-25。
- **入库日期：** 2026-10-09
- **一句话说明：** 给定多视角初始图像、相机标定、机器人 MJCF/URDF 与外部策略提供的未来关节轨迹，Uranus 以自回归视频扩散预测机器人执行动作后的同步多视角画面；它是可交互的视觉模拟器，不是负责输出控制动作或替代物理引擎的策略/动力学求解器。

## 开源状态（2026-10-09 核查）

| 组件 | 状态 |
|------|------|
| 论文、项目页 | 已公开 |
| 推理代码 | 官方仓库 Uranus-OSS 已公开，Apache-2.0 |
| 演示数据 | Hugging Face：<https://huggingface.co/datasets/D-Robotics/Uranus-Demo-Data> |
| 权重 | Hugging Face 集合：<https://huggingface.co/collections/D-Robotics/uranus>；包含 Uranus-1.3B 与蒸馏版 |
| 训练代码/全量训练数据 | 不应由“推理代码和 demo 已公开”推断为完整开放；论文/仓库公开范围以各自说明为准 |
| SDK | 官方代码 README 链接另有 Uranus-SDK；本条主要归档论文对应的 Uranus-OSS 推理入口 |

模型卡列出的 SFT 与蒸馏模型均为 384×640 推理配置；前者默认 25 步，蒸馏版 4 步。检查点约 17.5 GB。仓库 quickstart 使用 `uv` 环境，以 `main.py` 指定权重、样例输入和输出目录；样例元数据包含关节状态时间序列、机器人模型与相机/参考图像等条件。

## 核心方法

- **外层自回归滚动，内层 flow-matching latent diffusion：** 模型根据已生成历史和新动作条件逐步生成未来视频 latent，再经 VAE 解码为图像。
- **动作条件不是文本动作指令：** 输入是上游 policy 的未来关节位置轨迹（qpos）；结合机器人骨架/MJCF 或 URDF，以及相机标定生成几何条件。
- **多视角时空建模：** Plücker ray 特征编码相机几何；DiT 交替进行跨视角空间注意力和同一相机的时间注意力。一次 latent 预测解码为 4 帧同步 RGB，可在线反复滚动。
- **实时实现：** KV cache 复用历史，滑动窗口控制上下文；论文报告系统可达 24 FPS。论文所说的 3,300 小时数据和 64-GPU 训练属于作者报告，不等同于公开发布全量训练集。

## 评测与边界

论文以 WorldOlympiad 比较生成画面的交互、物理和 3D 一致性指标；作者报告 Uranus-1.3B 综合分 0.722。基准横向比较存在任务与调用粒度差异，不能把该分数当成与所有模拟器通用、完全同条件的排行榜结论。闭环策略实验报告 Uranus 评测与真实/干净环境趋势高度相关（Pearson 0.98），但真实一致性测试也显示明显泛化差距：论文给出的训练样本一致性为 86%，测试为 58%（各取样后由五位评审打分）。

**重要限制：** 它生成视觉后果，关节轨迹由外部控制器给出；不显式保证刚体约束、接触/抓取状态或任务状态正确。长时滚动会累积误差，分布外场景性能下降。因此应用于策略筛选/评测时仍需要物理仿真或真机复核。

**对 wiki 的映射：** [Uranus 项目节点](../../wiki/entities/paper-uranus.md)。
