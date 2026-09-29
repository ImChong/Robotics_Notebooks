# ULTRA（Sirui-Xu/ULTRA）

> 来源归档（ingest）

- **标题：** ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation
- **类型：** repo
- **组织：** Sirui Xu（UIUC）
- **链接：** <https://github.com/Sirui-Xu/ULTRA>
- **项目页：** <https://ultra-humanoid.github.io/>
- **论文：** <https://arxiv.org/abs/2603.03279>
- **入库日期：** 2026-09-29
- **一句话说明：** UIUC **IROS 2026** 统一多模态人形 loco-manipulation 官方实现：神经重定向 → privileged teacher PPO → 多模态 student 蒸馏 + RL finetune；Isaac Gym 训练、MuJoCo sim2sim、Unitree G1 部署与 TorchScript 导出。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-ultra-unified-multimodal-control-for-autonomous.md`](../../wiki/entities/paper-notebook-ultra-unified-multimodal-control-for-autonomous.md)

## 开放程度

| 项 | 状态 |
|----|------|
| 训练 / 推理代码 | **已开源**（Apache-2.0 + MIT InterMimic 栈） |
| 预置 teacher 权重 | 仓库内 `ultra/weights/teacher_ultra_inference.pth` |
| 运动数据 | Drive 链接（OMOMO 参考 + G1 增广轨迹）；需手动下载 |
| 依赖 | Isaac Gym Preview 4、Python 3.8、`requirements.txt`、MuJoCo（sim2sim） |

## 复现管线（README 五阶段）

| 阶段 | 入口 |
|------|------|
| 1 · 神经重定向 | `scripts/train_retarget_smplx.sh` |
| 2 · 导出增广轨迹 | `scripts/export_retarget_smplx.py` |
| 3 · Teacher 跟踪 | `scripts/train_teacher.sh` |
| 4 · Student 蒸馏 | `scripts/train_student.sh` / `train_student_multigpu.sh` |
| 5 · RL finetune | `scripts/train_finetune.sh <student.pth>` |

**推理：** `ultra/run_teacher_inference.py`；`scripts/play_student.sh`、`sim2sim_student.sh`、`export_jit.sh`。

## 对 wiki 的映射

- 论文 source：[ultra_arxiv_2603_03279.md](../papers/ultra_arxiv_2603_03279.md)
- 项目页：[ultra-humanoid-github-io.md](../sites/ultra-humanoid-github-io.md)
- 同系列：[InterMimic](https://github.com/Sirui-Xu/InterMimic) / [InterPrior](https://github.com/Sirui-Xu/InterPrior)（HOI 生成控制谱系）
