# muscle-supermimic

> 来源归档

- **类型：** repo
- **链接：** <https://github.com/kaibohex/muscle-supermimic>
- **项目页：** <https://muscle-supermimic.beanpow.top/>（[归档](../sites/muscle-supermimic.md)）
- **许可证：** 代码 Apache-2.0；依赖模型与动作数据另有使用限制
- **入库日期：** 2026-10-27
- **一句话说明：** 在 mjlab / MuJoCo-Warp 上把 SuperMimic 风格训练用于肌骨人体的全身跟踪。
- **沉淀到 wiki：** [muscle-supermimic](../../wiki/entities/muscle-supermimic.md)

## 复现入口和边界

README 提供 `uv sync`、`train`、`ms-retarget-myofullbody`、`ms-eval`、`ms-render` 命令，MS-Human-700 为子模块。默认 4096 环境下，作者称 MyoFullBody 需 24 GB 显存，MS-Human-700 需 32 GB；较小配置参见训练文档。动作来自受限访问的 MuscleMimic 重定向 AMASS KIT 数据，需接受数据条款；AMASS 数据及其重定向结果不可再分发，且限非商业研究。MyoFullBody 所含 MoBL-ARMS 也仅限非商业用途。SuperMimic 实现移植自社区的 supermimic_vibed，并非原论文官方代码。
