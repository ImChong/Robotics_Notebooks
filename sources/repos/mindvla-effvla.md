# EFFVLA

- **标题：** EffVLA — action-head modeling code
- **类型：** repo
- **链接：** <https://github.com/mindvla-team/EFFVLA>
- **项目页：** <https://mindvla-team.github.io/EFFVLA/>
- **入库日期：** 2026-09-15
- **一句话说明：** EffVLA 官方 action-head 实现，供 starVLA 等框架 drop-in；含 EffVLA 与 Pi0 模块对照。
- **沉淀到 wiki：** [`wiki/entities/paper-effvla.md`](../../wiki/entities/paper-effvla.md)
- **交叉归档：** [`sources/sites/effvla-mindvla-github-io.md`](../sites/effvla-mindvla-github-io.md)

## 复现路径（README 级）

1. Clone 仓库并安装依赖（对齐 starVLA / 自有 VLA 栈）。
2. 接入 SigLIP2 + Qwen2.5 骨干与 KV-share 接口。
3. 使用 **VLM-init** 4-layer transformer head + 单 pass L1（EffVLA 配方）。
4. 在 LIBERO / LIBERO-Plus 或自有 DROID 管线评测延迟与成功率。

## 开源边界

- **已发布：** action-head 建模与模块代码。
- **未见：** 完整训练脚本、checkpoint、DROID 数据管线一键复现。
