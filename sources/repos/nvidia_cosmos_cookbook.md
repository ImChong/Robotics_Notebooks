# nvidia-cosmos/cosmos-cookbook

> 来源归档

- **标题：** Cosmos Cookbook
- **类型：** repo
- **组织：** nvidia-cosmos
- **代码：** <https://github.com/nvidia-cosmos/cosmos-cookbook>
- **文档站：** <https://nvidia-cosmos.github.io/cosmos-cookbook/>
- **Stars：** ~471（2026-09-05）
- **入库日期：** 2026-09-05
- **一句话说明：** Cosmos 生态 **后训练脚本 + 案例** 仓：`docs/` 写配方，`scripts/` 跑策展 / 评测 / 配置；GitHub Pages 即 Cookbook 站。
- **沉淀到 wiki：** 是 → [`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（代码 Apache-2.0；权重走各模型卡） |
| **代码** | <https://github.com/nvidia-cosmos/cosmos-cookbook> |
| **文档** | <https://nvidia-cosmos.github.io/cosmos-cookbook/> |
| **维护** | README **Important**：不再积极开发，仅有限维护；新功能聚焦 [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) |

## README 要点（2026-09-05）

- 最新配方表（至 2026-04-21）：农机 Transfer 2.5 后训练、Reason2@Jetson Thor、Beamr CABR 策展、手术 Predict 2.5、embedding 轨迹离群检测、GR00T-Dreams、Cosmos Policy 升 Predict 2.5。
- 本地文档：`just install` / `just serve-external`；配方 GPU 工作流仅 Ubuntu。
- 媒体走 Git LFS；贡献指南要求优先 `.mp4`（H.264）而非 GIF。

## 对 wiki 的映射

- 实体：[`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)
- 站点摘录：[`sources/sites/cosmos-cookbook.md`](../sites/cosmos-cookbook.md)
- Transfer：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)

## 官方原文复核（2026-11-17）

- **版本：** `d0857364e8a727be41b181731e03f478213e4558`。
- **核对范围：** 官方 README 全文与 justfile（install、模板生成、serve-external 和 CI 构建）；未运行 GPU 训练、推理或真机实验。
- **通读：** 已完成

README 仍将旧 Cookbook 标为有限维护并引导迁移 NVIDIA/cosmos。文档入口 just serve-external 先运行 makejinja 模板（external.yaml），再 uv run mkdocs serve；这只是配方文档运行，不会自动下载所有模型或启动 GPU 推理。
