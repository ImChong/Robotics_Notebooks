# urdd_beyond_urdf_arxiv_2512_23135

> 来源归档（ingest）

- **标题：** Beyond URDF: The Universal Robot Description Directory for Shared, Extensible, and Standardized Robot Models
- **类型：** paper
- **来源：** arXiv:2512.23135（HTML v2；PDF <https://arxiv.org/pdf/2512.23135>）
- **作者：** Roshan Klein-Seetharaman、Daniel Rakita（Yale / APOLLO Lab 语境）
- **入库日期：** 2026-05-17
- **一句话说明：** 提出 **URDD（Universal Robot Description Directory）**：在保留 URDF 等原始规格的前提下，把各框架反复从 URDF **派生** 的几何/拓扑/DOF 映射等结果，拆成 **版本化、可增量扩展** 的 JSON/YAML **模块目录**；配套 **Rust（含 Bevy 检视）** 与 **Three.js 浏览器检视器**，并论证其相对单文件规格与「各栈各算一遍」的工程冗余问题。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2512.23135>
- **核心贡献：** 传统 URDF/SDF/MJCF/USD 多只承载 **最小原始** 运动学、动力学与几何引用；仿真、规划、控制、可视化各自 **重复推导** DOF 计数、链结构、凸分解、包围体等 → 冗余实现与弱标准化。URDD 把这些 **派生信息模块化落盘**，以目录形式组织，目标成为跨框架 **共享预处理层**。
- **对 wiki 的映射：**
  - [URDD 实体](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)
  - [Robot Viewer](../../wiki/entities/robot-viewer.md)（多格式 Web 检视对照）
  - [URDF-Studio](../../wiki/entities/urdf-studio.md)（上游 URDF 编辑与多格式导出）

### 2) 与 URDF+ / Extended URDF / RobCoGen 等的关系（Related Work）

- **链接：** <https://arxiv.org/html/2512.23135v2#S2>
- **核心贡献：** URDF+、Extended URDF 等 **扩展 XML 规格本身**；URDD **不替代** 输入规格，而是 **与输入格式正交**：从 URDF（或未来其他格式）生成 **预处理后的派生数据**，以 JSON/YAML 模块供任意语言消费；与 RobCoGen 等 **代码生成** 路线对比：URDD 偏 **语言无关数据资产**，RobCoGen 偏 **可执行专用代码**。
- **对 wiki 的映射：**
  - [Pinocchio](../../wiki/entities/pinocchio.md)（常见「从模型描述推导动力学/几何」栈）
  - [MuJoCo](../../wiki/entities/mujoco.md)（MJCF 作为另一套仿真侧描述）

### 3) 目录结构与代表性模块（Sec. III）

- **链接：** <https://arxiv.org/html/2512.23135v2#S3>
- **核心贡献：** 顶层 **metadata**（机器人名、版本）；文内称当前实现约 **15 个模块**（完整列表见作者公开文档）。代表模块包括：**URDF 模块**（原 URDF 信息转存为 JSON/Yaml）、**DOF 模块**（DOF 数与关节索引↔DOF 向量映射）、**Connections**（任意两 link 间路径）、**Chain**（父子层次，支撑 FK 构造）、**Bounds**（关节上下限）、**Mesh 模块族**（原网格与凸包/凸分解等派生网格及路径）、**Link Shapes**（OBB/包围球等近似 + 自碰撞跳过元数据等）。**每模块独立版本号**，新模块可加入而不破坏旧解析路径。
- **对 wiki 的映射：**
  - [paper-urdd-universal-robot-description-directory](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)

### 4) 工具链与评测要点（Sec. IV–V）

- **链接：** <https://arxiv.org/html/2512.23135v2#S4>、<https://arxiv.org/html/2512.23135v2#S5>
- **核心贡献：** **Rust** 转换器：URDF → 全量模块 + 多格式网格导出；**Bevy** GUI 用于检视网格/凸分解/包围体并交互编辑 **自碰撞跳过对**；支持 **多 URDD 组合**（夹爪+臂+底盘）。**Three.js** 浏览器检视器直接读目录。**计时**：在论文给出的笔记本配置下，UR5 / XArm7 / Unitree B1 / Orca hand / Unitree H1 等 **秒级～约 90 s** 量级完成生成（与网格复杂度相关）。**体积**：URDD（含派生网格）显著大于纯 URDF，但论文强调换来的是 **可复用派生数据**；**FK 实验**：展示用轻量解析即可基于模块拼装 FK。
- **对 wiki 的映射：**
  - [apollo-resources 归档](../repos/apollo-lab-yale-apollo-resources.md)
  - [浏览器检视站点](../sites/apollo-lab-yale-apollo-resources-github-io.md)

## 其他公开资料（非 PDF 正文）

- **浏览器内可视化（读 `apollo-resources` 仓内 URDD 资产）：** <https://apollo-lab-yale.github.io/apollo-resources/> — 站点归档见 [sources/sites/apollo-lab-yale-apollo-resources-github-io.md](../sites/apollo-lab-yale-apollo-resources-github-io.md)
- **Rust 实现与示例 URDD：** <https://github.com/Apollo-Lab-Yale/apollo-rust> — 归档见 [sources/repos/apollo-lab-yale-apollo-rust.md](../repos/apollo-lab-yale-apollo-rust.md)
- **Three.js 引擎模块（import map 引用 `apollo-three-engine`）：** <https://github.com/Apollo-Lab-Yale/apollo-three-engine> — 归档见 [sources/repos/apollo-lab-yale-apollo-three-engine.md](../repos/apollo-lab-yale-apollo-three-engine.md)
- **Python 包骨架：** <https://github.com/Apollo-Lab-Yale/apollo-py> — 归档见 [sources/repos/apollo-lab-yale-apollo-py.md](../repos/apollo-lab-yale-apollo-py.md)

## 当前提炼状态

- [x] 论文摘要与核心方法摘录
- [x] wiki 页面映射与姊妹资料互链
