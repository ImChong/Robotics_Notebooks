# Exercises Dataset（hasaneyldrm/exercises-dataset）

> 来源归档

- **标题：** Exercises Dataset — 1,324 条健身动作目录（元数据 + 多语说明 + 180×180 媒体）
- **类型：** repo / dataset
- **来源：** Hasan Emir Yıldırım（GitHub: `hasaneyldrm`）
- **链接：** https://github.com/hasaneyldrm/exercises-dataset
- **配套应用：** [LogPress](https://github.com/hasaneyldrm/logpress-public)（AI 辅助健身记录；本仓为其 exercise data layer）
- **入库日期：** 2026-07-23
- **一句话说明：** 面向健身 App / 推荐 / 识别原型的 **结构化动作目录**：1,324 条记录含部位、器械、目标肌群、10 语说明与逐步步骤，并附 180×180 缩略图与动画 GIF（媒体 © Gym visual，非 MIT）。
- **开源状态：** **已开源（分层许可）** — 代码、JSON 结构与说明文本为 **MIT**；`images/`、`videos/` 媒体 © [Gym visual](https://gymvisual.com/)，以仓库 `LICENSE` 媒体例外 + `NOTICE.md` 为准，克隆仓库 **不** 自动授予媒体商用/再分发权。
- **沉淀到 wiki：** 是 → [`wiki/entities/exercises-dataset.md`](../../wiki/entities/exercises-dataset.md)

## 为何值得保留

- **易与人形 MoCap 混淆的公开「exercise dataset」**：名称与星标易被误读为可重定向的参考运动；入库后可明确边界——**无 BVH / SMPL / 关节角序列**，不能直接喂 WBT / AMP。
- **结构化标签层**：`body_part` / `equipment` / `target` / `muscle_group` / `secondary_muscles` + 10 语 `instructions` / `instruction_steps`，适合健身动作分类、推荐或 **文本–视觉** 原型，而非物理跟踪。
- **工程可复制**：`data/exercises.json` + JSON Schema、`index.html` 浏览器、`setup.html` 多库 SQL / API 脚手架；LogPress 为下游消费样板。

## 数据与许可要点（编译自 README / LICENSE / NOTICE）

| 项 | 内容 |
|----|------|
| 规模 | **1,324** 条；每条 1 缩略图 + 1 动画 GIF（180×180） |
| 主文件 | `data/exercises.json`、`data/exercises.schema.json` |
| 部位分布（README） | Upper Arms 292 · Upper Legs 227 · Back 203 · Waist 169 · Chest 163 · Shoulders 143 · Lower Legs 59 · Lower Arms 37 · Cardio 29 · Neck 2 |
| 器械 Top | Body Weight 325 · Dumbbell 294 · Cable 157 · Barbell 154 · …（约 25% 纯自重） |
| 语言 | en / es / it / tr / ru / zh / hi / pl / ko / fr |
| 媒体权利人 | Gym visual；须保留 `© Gym visual — https://gymvisual.com/`；分辨率限制 180×180 |
| Stars（入库时） | ~16.6k（GitHub） |

## 对 wiki 的映射

- [Exercises Dataset 实体页](../../wiki/entities/exercises-dataset.md) — 定位、schema、与 AMASS/LaFAN1 等选型对照
- [人形参考运动数据集选型](../../wiki/comparisons/humanoid-reference-motion-datasets.md) — 「常见误区」：勿当 MoCap 源
