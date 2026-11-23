# Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System

> 来源归档（论文原文与项目页）

- **标题：** Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System
- **类型：** paper + project-site
- **arXiv：** <https://arxiv.org/abs/2604.24921>（HTML：<https://arxiv.org/html/2604.24921>）
- **项目页：** <https://libra-vla.github.io/>
- **代码：** 项目页标注 **Code (coming soon)**（截至 2026-11-21 无公开 GitHub 仓库）
- **作者：** Yifei Wei, Linqing Zhong, Yi Liu, Yuxiang Lu, Xindong He, Maoqing Yao, Guanghui Ren（北航、AgiBot；ACL 2026）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 arXiv HTML 全文（<https://arxiv.org/html/2604.24921>，§1–§5 与 Limitations，含 Hybrid Action Space 粗细解耦、System 2 并行粗动作头、System 1 扩散动作细化器、FIFO Intent Buffer 异步执行策略与 LIBERO/真机实验）及项目页 <https://libra-vla.github.io/>；未运行代码。
- **通读：** 已完成
- **一句话说明：** 北航与智元提出的异步粗到细双系统 VLA 框架：将动作分解为离散宏观意图（System 2 InternVL2.5-2B + 并行粗动作头，低频预测扩展视界并写入 FIFO Intent Buffer）与连续微调动作（System 1 独立 SigLIP 视觉编码 + 扩散细化器，高频弹出执行），以“Libra Point”均衡双阶段学习复杂度。

## 核心论文要点（通读提炼）

1. **混合动作空间解耦（§3.1）：** $P(\mathbf{a}_t \mid \mathbf{o}_t, L) \approx P(\mathbf{a}_t^f \mid \mathbf{a}_t^c, \mathbf{o}_t) \cdot P(\mathbf{a}_t^c \mid \mathbf{o}_t, L)$，用粗粒度离散桶（$N \ll 256$）对齐 VLM 语义推理，用连续扩散精修几何微调。
2. **System 2 & System 1（§3.2–3.3）：** System 2（InternVL2.5-2B + 12 层双向并行粗动作头）一次输出 $L_{\text{macro}} = M \times H_{\text{chunk}}$ 步粗意图；System 1 配备独立视觉编码器（SigLIP）+ 12 层扩散细化器，训练期随 System 2 准确率从真值粗桶过渡到预测分布采样（动态课程学习）以培养抗噪纠偏能力。
3. **FIFO Intent Buffer 异步执行（§3.4）：** System 2 低频补满队列，System 1 在随后 $M-1$ 个控制周期仅弹出对应切片生成精细动作，消除高频步上的 VLM 阻塞。
4. **实验与局限（§4–5, Limitations）：** LIBERO 均分 **97.2%**（Long **92.8%**）；主要局限在于当前异步队列按序消耗全部粗意图、缺乏实时置信度重规划门控，异步频率过高时性能会轻度衰减；代码尚未开源。

## 对 wiki 的映射

- [GO-2（智元执行基座）](../../wiki/entities/go-2.md)
