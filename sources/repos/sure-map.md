# SURE-Map

- **标题**: SURE-Map: Self-Correcting Streaming Geometric Foundation Models
- **链接**: [https://github.com/RCL-Robotics/SURE-map](https://github.com/RCL-Robotics/SURE-map)
- **类型**: repo
- **作者**: Liu, Mingkai; Zhao, Hao; Zuo, Xingxing (2026)
- **摘要**: 在 **VGGT 系流式几何基础模型**（论文实验骨干为 **LingBot-Map**）上叠加 **自校正** 模块：**跨视几何不确定性头**（冻结骨干、TartanAir 训练）与 **多时间尺度推理**（快因果流 + 稀疏 keyframe-window 尺度重标定）。提供 `online/` 长程位姿评测、`benchmark/` 室内点云与 DTU 管线；依赖 **FlashInfer** paged KV；**torch 2.8.0 + cu128** 为 README 推荐栈。
- **项目页**: [mingkai-liu.github.io/projects/sure-map](https://mingkai-liu.github.io/projects/sure-map/)
- **权重**: [milchstrasse/SURE-Map](https://huggingface.co/milchstrasse/SURE-Map)（uncertainty）；骨干 [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map)

## 核心要点

1. **模块化：** 不确定性 checkpoint 仅含 **uncertainty head**；复现论文需额外下载 **lingbot-map.pt** 作骨干。
2. **训练：** `train_flow_sigma_tartanair.py`，8×GPU，8–24 帧 clip，`518×392`；TartanAir forward flow 与推理 backward-flow 记号在 README 有说明。
3. **长程位姿：** `online/run_kitti.py`、`run_oxford.py`、`run_vbr.py` + 对应 YAML。
4. **室内重建：** `benchmark/prepare.py` + `run.py` + `eval_*_point_uncertainty.py`（NRGBD / 7-Scenes / DTU）。
5. **许可：** 主体 **Apache-2.0**；部分 DUSt3R 衍生训练代码 **CC BY-NC-SA 4.0**。
6. **致谢依赖：** VGGT、DINOv2、FlashInfer、**LingBot-Map**。

## 为什么值得保留

- 把「流式几何 FM 会漂移」问题拆成 **可学习的跨视一致性不确定性** + **显式多尺度校正**，工程上与 LingBot-Map 栈对齐，便于机器人长视频几何模块选型。
- 评测脚本覆盖 **KITTI / Oxford Spires / VBR** 等长程户外与室内点云，与 [LingBot-Map](../../wiki/entities/paper-lingbot-map.md) 形成直接对照。

## 对 wiki 的映射

- [wiki/entities/paper-sure-map.md](../../wiki/entities/paper-sure-map.md): 论文实体页
- [wiki/entities/paper-lingbot-map.md](../../wiki/entities/paper-lingbot-map.md): 骨干与基线对照
- [wiki/methods/lingbot-map.md](../../wiki/methods/lingbot-map.md): 上游流式几何 FM

---
- **录入日期**: 2026-09-24
