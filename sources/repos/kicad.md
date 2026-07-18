# KiCad（官方源码仓库）

> 来源归档

- **标题：** KiCad EDA — 官方源码
- **类型：** repo
- **来源：** KiCad 项目
- **主仓库（开发真源）：** https://gitlab.com/kicad/code/kicad
- **GitHub 镜像：** https://github.com/KiCad/kicad-source-mirror（只读同步，**不接受 PR**）
- **官网：** https://www.kicad.org/
- **文档：** https://docs.kicad.org/
- **入库日期：** 2026-07-18
- **一句话说明：** KiCad 桌面 EDA 套件的 **C++/wxWidgets** 官方源码；**GPLv3** 为主许可，含多组件许可文件；GitLab 为贡献入口，GitHub 为便于 star/issue 浏览的镜像。
- **沉淀到 wiki：** [KiCad](../../wiki/entities/kicad.md)

---

## 开源状态（2026-07-18 项目页核查）

| 项 | 状态 |
|----|------|
| **主程序源码** | **已开源** — GitLab `kicad/code/kicad`，默认分支 `master`；稳定分支 `10.0`、`9.0` 等 |
| **贡献流程** | Merge Request 在 **GitLab**；见仓库 `CONTRIBUTING.md` |
| **GitHub 镜像** | README 明确：*Pull requests on GitHub are not accepted or watched* |
| **许可** | 根目录 `LICENSE`（GPLv3）；`LICENSE.*` 列出 BSD-3、Apache-2.0、MIT、CC0 等第三方组件 |
| **预编译包** | 官网下载安装包；非仅源码 |

---

## 仓库结构（README / 目录摘要）

| 子目录 | 说明 |
|--------|------|
| `eeschema/` | 原理图编辑器 |
| `pcbnew/` | PCB 编辑器 |
| `3d-viewer/` | 3D 查看器 |
| `gerbview/` | Gerber 查看器 |
| `pcb_calculator/` | PCB 工程计算器 |
| `kicad/` | 项目管理器 |
| `cvpcb/` | 封装关联工具 |
| `common/`, `libs/`, `include/` | 公共库与几何工具 |
| `qa/` | 单元测试 |
| `translation/` | 翻译（多数语言经 Weblate） |
| `api/` | 脚本/API 相关（版本演进中） |

**构建：** CMake + wxWidgets 3.2+；`INSTALL.txt`、`install-deps.sh`；CI 覆盖 Linux/Windows（GitLab pipeline）。

**近期标签（摘录）：** `10.0.4`、`10.0.5-rc1`、`10.0.0` 等。

---

## 与机器人研究/工程的关联点

- **自研关节驱动板**：在 GitLab 跟踪 KiCad 版本与插件兼容性；fork 官方库符号/封装或引用 [kicad-kicad](https://gitlab.com/kicad/libraries/kicad-symbols) 等官方库项目。
- **与固件栈分工**：KiCad 交付 **netlist + layout + BOM**；[SimpleFOC](../../wiki/entities/simplefoc.md) / STM32 固件在另一仓库；二者通过 **引脚定义与 ADC 通道分配表** 对齐。
- **与机械 CAD**：STEP 外壳可导入 3D 查看器做 **板卡机械干涉** 检查；机械真值仍在 [FreeCAD](../../wiki/entities/freecad.md) 等。

---

## 交叉链接

- 项目页归档：[kicad-org.md](../sites/kicad-org.md)
- 用户文档归档：[kicad_docs_10_zh.md](../courses/kicad_docs_10_zh.md)
- Wiki 实体：[kicad.md](../../wiki/entities/kicad.md)
