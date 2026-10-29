# FlowHMR: Physically Plausible Motion Capture from Video

- **类型：** paper
- **链接：** <https://arxiv.org/abs/2610.03691>
- **入库日期：** 2026-10-28
- **机构：** 北京大学、香港科技大学、腾讯、华东师范大学、中山大学、浙江大学

## 资料要点

- 单目视频经条件流匹配生成全局人体、手部及轨迹，再以物理跟踪奖励进行 GRPO 后训练。
- Wild-4K 上 PHC+ 仿真跟踪成功率从 base 的 78.79% 提升至 82.47%；这是作者报告的控制器跟踪指标，不等于真机成功率。
- 官方公开推理、训练和权重；训练视频不随仓库提供，Wild-4K 发布片段清单而非视频。SMPL 系列、VGGT-Omega、SAM 3D Body 等资源需单独处理许可或访问。

## 对 wiki 的映射

- [论文实体](../../wiki/entities/paper-flowhmr.md)：方法、实验及工程局限。
