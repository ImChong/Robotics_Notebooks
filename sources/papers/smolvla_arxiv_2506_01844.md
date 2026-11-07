# SmolVLA（arXiv:2506.01844）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2506.01844>
- **PDF：** <https://arxiv.org/pdf/2506.01844>
- **作者：** Shukor, Aubakirova, Capuano, Kooijmans, Palma, Zouitine 等（arXiv 摘要页前 6 位）
- **提交：** 2025-06-02
- **代码：** <https://github.com/huggingface/lerobot>（git ls-remote 可达）
- **入库日期：** 2026-11-07（首次入库 2026-11-02）
- **版本：** arXiv:2506.01844（2025-06-02 提交；本次核对的 PDF 为 2026-11-07 下载版）
- **核对范围：** 2026-11-07 下载 PDF（24 页）并读完摘要、§1 引言、§2 相关工作、§3 方法（架构、社区数据预训练、异步推理）、§4 实验（设置、实现细节、基线、主结果、异步推理、全部消融 Table 6–13）、§5 讨论与局限；附录 A 的 481 个社区数据集清单与任务标注提示词只浏览，未逐项核对；未运行 LeRobot 代码。
- **通读：** 已完成
- **线索来源：** [x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md](../blogs/x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md)

## arXiv 摘要要点（人工转述）

论文摘要称：SmolVLA 是 small、efficient、community-driven 的 VLA，大幅降低训练与推理成本且保持有竞争力的性能；设计为可单 GPU 训练、消费级 GPU 或 CPU 部署；引入异步推理栈，使感知与动作预测与动作执行解耦，配合 chunked 动作生成获得更高控制率；性能可比 10 倍大的 VLA；并开源全部代码、预训练模型与训练数据。

## 对 wiki 的映射

- [paper-smolvla](../../wiki/entities/paper-smolvla.md)

## 通读核对结论（2026-11-07）

- **结构：** SmolVLM-2（SigLIP + SmolLM2）作 VLM；不做图像切块、每帧限制 64 个视觉 token；只用前 N=L/2 层（实现里取前 16 层）；动作专家 $v_\theta$ 隐层宽度为 VLM 的 0.75×，主模型 450M 参数、其中动作专家约 100M；动作块 n=50，推理固定 10 步 flow matching。
- **注意力：** 动作专家交错使用交叉注意力（K/V 来自 VLM 特征）与对动作 token 的**因果**自注意力。消融（LIBERO）：CA 79.0、SA 74.5、交错 CA+SA 85.5；动作 token 间双向注意力 67.5，低于因果 74.5（与 π0 动作专家全双向的做法不同）。
- **目标：** 沿用 $A^\tau=\tau A+(1-\tau)\epsilon$，$\tau$ 取 Beta 分布；论文文字把目标场写成 $u=\epsilon-A$，而按该插值式对 $\tau$ 求导应为 $A-\epsilon$，二者符号相反（原文自身不一致，实现须以代码为准）。flow matching 显著优于 L1 回归（Table 10）。
- **数据与训练：** 481 个社区数据集、约 22.9K 条轨迹、10.6M 帧（论文称“少于 30k episodes”）；用 Qwen2.5-VL-3B 重写任务描述；预训练 200K 步、batch 256；约 30k GPU 小时。
- **结果：** LIBERO 平均：SmolVLA 0.45B 为 87.3，π0（3.3B，机器人预训练）86.0，π0（PaliGemma 初始化）71.8；Meta-World 平均 57.3（0.45B）。真机 SO100（多任务）：SmolVLA 78.3，π0 61.7，ACT 48.3；预训练带来 51.7 → 78.3。异步推理：成功率相近，单次 pick-place 平均 9.7 s（同步 13.75 s），60 s 内完成 19 次（同步 9 次）。
- **异步推理（§3.3）：** 队列剩余比例低于阈值 g 时触发新推理；$g\ge E[\ell_S]/(n\Delta t)$ 可避免队列耗尽；用关节空间相似度过滤近重复观测；新旧队列在重叠段聚合（论文未给聚合细节，不同于 RTC 的 inpainting）。
- **原文自述局限：** 预训练只用单一机器人（SO100）数据、约 23K 轨迹，远少于 OpenVLA 的约 100 万条；VLM 骨干偏文档/OCR；任务较简单、短视界；主要为模仿学习。
