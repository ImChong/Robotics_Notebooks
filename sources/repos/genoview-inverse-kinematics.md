# GenoView-InverseKinematics

> 来源归档（ingest）

- **标题：** GenoView-InverseKinematics
- **类型：** repo
- **链接：** https://github.com/orangeduck/GenoView-InverseKinematics
- **入库日期：** 2026-09-13
- **许可证：** MIT
- **一句话说明：** theorangeduck 博文《Inverse Kinematics and Foot Locking》的配套 raylib 示例：GenoView 骨架动画查看器 + 两骨 IK、运行时惯性化足锁、接触标注与离线 PBD 式脚滑移除的可运行 C 实现。
- **开源状态：** **已开源**（MIT；README 含构建与 BVH 导出步骤）
- **配套文章：** https://theorangeduck.com/page/inverse-kinematics-foot-locking — [`sources/blogs/orangeduck_inverse_kinematics_foot_locking.md`](../blogs/orangeduck_inverse_kinematics_foot_locking.md)
- **沉淀到 wiki：** 是 → [`wiki/entities/genoview-inverse-kinematics.md`](../../wiki/entities/genoview-inverse-kinematics.md)

## 为什么值得保留

- **动画/重定向后处理的可复现代码**：腿链 IK、惯性化足锁、接触启发式、离线约束求解四段管线均有实现，比纯文字博文更易对照。
- **GenoView 查看器**：简易 deferred 渲染 + 网格地面，便于肉眼检查脚滑/穿透；支持 LaFAN1、ZeroEGGS 等 Geno 角色 BVH 导出脚本。
- **与机器人栈互补**：侧重 **游戏/动画 runtime** 足锁，可与 GMR/CoRe 等 **机器人重定向** 脚滑修补对照选型。

## 仓库要点（截至入库日）

| 项 | 内容 |
|----|------|
| 语言 / 依赖 | C + [raylib](https://www.raylib.com/) |
| 角色数据 | `resources/export_animations.py` 将 BVH 转为二进制；默认 Geno 骨架 |
| 核心入口 | `genoview.c` 加载动画；文内 `SolveLegChain`、`TwoBoneInverseKinematics`、`UpdateFootLockingState`、离线约束循环 |
| 纯 Python 版 | [GenoViewPython](https://github.com/orangeduck/GenoViewPython/)（本仓为 C/raylib 版） |

## 对 wiki 的映射

- [`wiki/entities/genoview-inverse-kinematics.md`](../../wiki/entities/genoview-inverse-kinematics.md)
- [`wiki/methods/foot-locking-ik-orangeduck.md`](../../wiki/methods/foot-locking-ik-orangeduck.md)
- [`wiki/formalizations/inverse-kinematics.md`](../../wiki/formalizations/inverse-kinematics.md)
