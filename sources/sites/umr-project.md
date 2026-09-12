# UMR 项目页（hanyang9.github.io/UMR）

> 来源归档

- **标题：** Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence
- **类型：** site（项目页 + 交互场景 + Studio 入口）
- **URL：**
  - 项目页：<https://hanyang9.github.io/UMR/umr_project.html>
  - UMR Studio：<https://hanyang9.github.io/UMR/umr_studio.html>
- **论文：** [arXiv:2609.02134](https://arxiv.org/abs/2609.02134)
- **代码：** [hanyang9/UMR](https://github.com/hanyang9/UMR)
- **机构：** 香港科技大学广州校区（HKUST-GZ）；诺亦腾机器人（Noitom Robotics）；汉阳大学（Hanyang University）；香港科技大学（HKUST）；香港大学（HKU）
- **入库日期：** 2026-09-12
- **一句话说明：** 官方页：多源/多机统一重定向概览、WebGL 交互场景、UMR Studio 浏览器 T-pose 与重定向体验、AdaPT body+racket 与真机部署视频。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Overview | 规范 T-pose 点云对应 + 约束重定向；接触图直传 |
| Unified Retargeting | 异构源 × 五台人形（0.75–1.83 m）交互场景 viewer |
| UMR Studio | 拖放 MJCF、调 T-pose、选参考动作；PiPlus / Adam / Fourier N1 等 preset |
| Morphologically Augmented | AdaPT G1+racket：SMPL-X+racket ↔ G1+racket 统一对应 |
| Real World Deployment | MimicKit / GRAIL / OmniContact 参考 + BeyondMimic / Holosoma / OmniContact 训练管线真机 |

## UMR Studio 要点

- 浏览器内完整 UMR 管线（采样 → 对应训练 → mesh 绑定 → 重定向）；**CPU 训练**，性能不代表原生实现。
- **许可限制：** 因数据许可，Studio **不提供重定向结果下载**；复现应走官方 GitHub。
- 内存建议：运行前保留 ≥ **3 GiB** 可用内存。
- 内置 T-pose 库：PiPlus、Booster K1、Unitree G1、EngineAI T800 等，可拖入 Studio 场景。

## 开源核查（步骤 2.5，2026-09-12）

| 资源 | 状态 |
|------|------|
| 官方 GitHub | **已开源** — [hanyang9/UMR](https://github.com/hanyang9/UMR)：训练/推理/批处理脚本、`robot_configs/`、多源 adapter（LAFAN1/SMPL-X、BONES-SEED、GRAIL、OmniContact、OMOMO、MimicKit、AdaPT、NR FBX/BVH） |
| UMR Studio | **已发布**（浏览器体验；结果不可下载） |
| SMPL-X 权重 | **不随仓分发**；需自行下载 neutral/male/female 模型 |
| OmniContact BVH 直读 | **部分**：内部 BVH→SMPL-X 转换器未公开；需预转换 SMPL-X 输入 |
| AdaPT 项目页「UMR coming soon」 | **已过时**（2026-09-12 核查）；官方仓与 Studio 已上线 |

## 对 wiki 的映射

- 主实体：[UMR 论文实体](../../wiki/entities/paper-umr-unified-motion-retargeting.md)
- 论文摘录：[umr_unified_motion_retargeting_arxiv_2609_02134.md](../papers/umr_unified_motion_retargeting_arxiv_2609_02134.md)
- 官方代码：[umr.md](../repos/umr.md)
- 非官方复现：[unified-motion-retargeting-unofficial.md](../repos/unified-motion-retargeting-unofficial.md)
