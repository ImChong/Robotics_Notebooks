# HY-WorldPlay / HY-World 1.5 — 官方主页与开源核验

- **名称：** HY-WorldPlay / HY-World 1.5
- **类型：** site / project
- **项目页：** <https://3d.hunyuan.tencent.com/sceneTo3D?tab=worldplay>
- **代码：** <https://github.com/Tencent-Hunyuan/HY-WorldPlay>
- **技术原文：** <https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf>
- **机构：** 腾讯混元（Tencent Hunyuan）
- **核对日期：** 2026-11-20
- **核对范围：** 官方仓库 README 链接、发布内容及对应介绍
- **通读：** 已完成

## 技术定位

WorldPlay 把键盘/相机轨迹转换成姿态条件，流式预测未来视频块；通过重构历史条件（reconstituted context memory）减少返回先前区域时的遗忘。仓库区分 HunyuanVideo 8B 路径（优先控制和记忆）与 WAN 5B 轻量路径。双向模型、因果模型与四步蒸馏模型提供不同的质量—时延取舍。

## 开放状态和复现提示

- 模型权重、推理和训练文档部分开放；个别模型变体仍待发布。
- README 的 RL+蒸馏联合权重仍列 To be released；使用者须区分仓库代码、已发布 checkpoint 与待发组件。`hyvideo/generate.py` 采用 Tencent Hunyuan Community License，不能擅自按 Apache 条款处理。项目技术报告出自 2025 年，不是 2026 年首次提出。
- 入口与源码分层见 [归档](../repos/hy-worldplay-1-5.md)。

## 映射

- [项目/论文详情](../../wiki/entities/hy-worldplay-1-5.md)
