# BeyondRetarget Plus（官方项目主页）

- **标题：** BeyondRetarget Plus: Data Engine for Generating Robot Motion Data from Internet-Scale Human Videos
- **类型：** site / project-page / monocular-video / humanoid-motion-data
- **URL：** <https://bear-ty.github.io/BR_plus_web/>
- **代码：** <https://github.com/bear-ty/BeyondRetarget_plus> — [源码归档](../repos/beyondretarget-plus.md)
- **基础版论文：** [BeyondRetarget（arXiv:2609.29850v1）](../papers/beyondretarget_arxiv_2609_29850.md)
- **基础版主页：** <https://bear-ty.github.io/Beyondretarget_page/>
- **交互演示：** <https://bear-ty-beyondretargetplus.hf.space/>
- **机构：** 南京大学电子科学与工程学院、智能科学与技术学院
- **入库日期：** 2026-11-22
- **核对范围：** 官方主页完整的摘要、Approach、Demo、Failure Case、To Be Continue、BibTeX，以及主页代码 / 演示入口；用 GitHub README 核对实际开源和复现边界。
- **通读：** 已完成
- **开源状态（2026-11-22）：** **已开源代码**；推理权重与机器人模型通过官方 Google Drive 提供；相机前端和 SMPL 等资源需按第三方条款另行获取。

## 来源要点

1. **任务定位：** 从互联网真实场景单目人体视频离线生成多种人形机器人的参考运动；与实时遥操作取向的 base 版不同。
2. **新增能力：** 大范围 global trajectory、移动相机支持；主页展示滑冰、侧翻、舞蹈、网球、上下楼梯等动态动作。
3. **方法说明：** 主页说明改进基础模型与轨迹估计、扩展训练集和机器人监督，但三张详细技术卡片目前仍标注 **Coming soon**；具体复现入口以 README / 源码为准。
4. **失败模式：** 均匀背景会削弱相机运动估计；运动模糊和分布外高动态动作也可能导致轨迹错误。
5. **未来方向：** 低时延单目 RGB 机器人控制尚为后续预告，不是 Plus 已公布的在线实时性能。

## 关联资料

- [BeyondRetarget Plus 官方源码归档](../repos/beyondretarget-plus.md)
- [BeyondRetarget 原论文归档](../papers/beyondretarget_arxiv_2609_29850.md)
- [BeyondRetarget 论文 / Plus 扩展实体](../../wiki/entities/paper-beyondretarget-monocular-humanoid.md)
