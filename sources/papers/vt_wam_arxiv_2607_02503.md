# VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation

- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2607.02503>
- **全文版本：** <https://arxiv.org/pdf/2607.02503v1>
- **核对日期：** 2026-11-13
- **核对范围：** arXiv v1 PDF，全文 §I–§V（含全部方法、表 I–III、图 1–6）；参考文献浏览，无单独附录。图表以文字层核对，未观看演示视频，未运行训练或真机实验。
- **通读：** 已完成
- **项目页：** <https://vt-wam.github.io/>

## 通读核对

- §III：三专家联合 flow matching；动作读首帧视觉及完整触觉去噪序列。部署 visual-cache 模式移除未来视觉，但仍生成未来触觉与动作，不是纯 action-only。
- AVTAG：接触由明显触觉形变识别；对视觉/触觉 keys stop-gradient，辅助 hinge max(0,pv−pt) 引导动作 query；λ=0.05，仅训练期启用。
- §IV：xArm7 + Robotiq 2F-85 + 双 Xense 35×20 形变场，腕相机 128×128，30 Hz 同步，每任务 100 示范，6 个单任务模型；每方法/任务 20 次。表面交互用 0/0.5/1 部分信用，插入任务用二值评分。
- 表 I：三表面任务均值 81.67%，三插入 61.67%，总均值 71.67%；与二值成功率的其他基准不可直接横比。
- 表 III：Asym(T0) 40/30，Asym(T Seq) 70/50，+AVTAG 85/55（擦花瓶/插管）；多任务与缩放律仍未验证。

## 对 wiki 的映射

- [VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)
