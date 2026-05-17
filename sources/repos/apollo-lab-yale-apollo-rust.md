# apollo-rust（Apollo-Lab-Yale）

> 来源归档

- **标题：** apollo-rust
- **类型：** repo（Rust；论文所述 URDF→URDD 转换与 Bevy 检视的主实现线索）
- **代码：** <https://github.com/Apollo-Lab-Yale/apollo-rust>
- **入库日期：** 2026-05-17
- **一句话说明：** 公开仓库根含 **`Cargo.toml`、`crates/`、`ur5_urdd/`** 等：对应论文 **Rust 预处理器 + 示例 URDD 输出** 的工程落点；具体 README 以仓库当前版本为准。
- **沉淀到 wiki：** [URDD 论文实体](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [URDD 论文实体](../../wiki/entities/paper-urdd-universal-robot-description-directory.md) | 解释 **为何需要** 与 **模块边界**；本仓给出 **可克隆的实现入口** |
| [Pinocchio](../../wiki/entities/pinocchio.md) | 传统栈多从 URDF 在线推导；URDD 尝试把高频派生结果 **固化为可交换资产** |

---

## 对 wiki 的映射

- [`wiki/entities/paper-urdd-universal-robot-description-directory.md`](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)
- [`sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md`](../papers/urdd_beyond_urdf_arxiv_2512_23135.md)
