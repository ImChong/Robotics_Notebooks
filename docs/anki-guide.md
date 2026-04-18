# Anki 闪卡使用指南

本 wiki 支持一键导出 Anki 兼容闪卡，帮助你在学习路线推进过程中巩固核心概念。

---

## 导出闪卡

```bash
# 导出全部闪卡（formalization + concepts）
make anki

# 仅导出公式推导类（形式化定义）
python3 scripts/export_anki.py --deck formalization

# 仅导出控制稳定性相关（CLF/CBF/Lyapunov/LQR 等）
python3 scripts/export_anki.py --deck control-stability

# 仅导出核心概念卡（有"一句话定义"的 concepts 页）
python3 scripts/export_anki.py --deck concepts-core
```

输出文件位于 `exports/` 目录：
- `anki-flashcards.tsv` — 全量卡
- `anki-formalization.tsv` — 公式推导卡
- `anki-control-stability.tsv` — 控制稳定性卡

---

## 导入 Anki

1. 打开 Anki → **文件 → 导入**
2. 选择 `exports/anki-*.tsv` 文件
3. 导入设置：
   - **类型**：基础（正反面）
   - **牌组**：新建牌组，如 `Robotics::Formalization`
   - **字段映射**：第 1 列 → 正面，第 2 列 → 背面，第 3 列 → 标签
   - **分隔符**：制表符（Tab）
4. 点击**导入**

---

## 建立子牌组

每张卡片的 Tags 字段格式为 `robotics::formalization::lyapunov`，Anki 会自动按 `::` 建立层级子牌组。

推荐牌组结构：

```
Robotics
├── formalization   # 公式推导
├── concept         # 概念定义
└── control-stability # 控制稳定性专题
```

---

## 配合路线学习

| 学习阶段 | 推荐牌组 | 搭配页面 |
|---------|---------|---------|
| 控制理论入门 | `control-stability` | [LQR](../wiki/formalizations/lqr.md) · [Lyapunov](../wiki/formalizations/lyapunov.md) |
| 强化学习基础 | `formalization` | [MDP](../wiki/formalizations/mdp.md) · [Bellman](../wiki/formalizations/bellman-equation.md) · [GAE](../wiki/formalizations/gae.md) |
| 全身控制实践 | `concepts-core` | [WBC](../wiki/concepts/whole-body-control.md) · [TSID](../wiki/concepts/tsid.md) |

每次学习完一个 wiki 页面后，运行 `make anki` 更新卡片再复习，效果最好。

---

## 更新卡片

wiki 更新后，重新导出并在 Anki 中**导入同名文件**——Anki 会自动根据 Tags 去重更新，不会重复添加。
