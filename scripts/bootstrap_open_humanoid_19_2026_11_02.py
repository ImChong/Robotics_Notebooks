#!/usr/bin/env python3
"""Bootstrap ingest: GitHubStore《19款能自己造的开源人形机器人，附完整开源地址》（2026-11-02）。

- 抓取：`scripts/fetch_wechat_article.py`。文中 19 个项目 → 19 个唯一 `wiki/entities/` 节点：
  新建 13，复用 6（Otto / Qmini / Open Duck Mini / MicroDuck / Microban / InMoov）。
- 核查（入库日）：GitHub 仓库 `git ls-remote` + 浅克隆 README（只读）；
  无 GitHub 仓库的项目（Nova、Rapiro 官方、InMoov 官网）只记文章表述，标「未核实」。
- 用法：python3 scripts/bootstrap_open_humanoid_19_2026_11_02.py <抓取的 txt>
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
URL = "https://mp.weixin.qq.com/s/UUwUAcKdVrlo-N60Oar3Xg"
BLOG = "wechat_githubstore_19_buildable_open_source_humanoids_2026-11-02.md"
MAP = "open-source-buildable-humanoids-19-technology-map.md"
MAP_TITLE = "能自己造的开源人形机器人 19 款：按成本与形态排序的阅读坐标"

HUM = "../entities/humanoid-robot.md"
OSH = "../entities/open-source-humanoid-hardware.md"
SIM = "../concepts/sim2real.md"
LOC = "../tasks/locomotion.md"

# 复用节点：序号、名称、节点、价格（文章）、备注
REUSE: list[dict[str, Any]] = [
    dict(
        n=2,
        name="Qmini",
        node="qmini",
        price="价格未确认",
        note="文章另列 RoboTamerSdk4Qmini 软件仓库（git ls-remote 可达，未建节点）",
    ),
    dict(n=4, name="Otto Humanoid / Otto DIY", node="otto-diy-biped", price="约 $50 起", note=""),
    dict(
        n=12,
        name="Open Duck Mini",
        node="open-duck-mini",
        price="低于 $400",
        note="Runtime 仓库在库内另有 open-duck-mini-runtime 节点",
    ),
    dict(
        n=13,
        name="MicroDuck",
        node="pollen-microduck",
        price="约 $399",
        note="RL 训练仓库在库内另有 pollen-microduck-rl 节点",
    ),
    dict(n=16, name="Microban", node="microban-humanoid", price="约 $550–600", note=""),
    dict(n=19, name="InMoov", node="inmoov-humanoid", price="上半身约 $900", note=""),
]

# 新建节点
NEW: list[dict[str, Any]] = [
    dict(
        n=1,
        slug="roboprime",
        name="RoboPrime",
        repo="simonepri/roboprime",
        price="约 $60–100",
        last="2017-10-19",
        lic="GPL-3.0（LICENSE 文件）",
        tags=["humanoid", "open-source", "3d-printing", "arduino", "education"],
        one="RoboPrime 是 simonepri 设计的 21 自由度全 3D 打印人形机器人，主控 ATmega328P（Arduino Pro Micro），搭配 21 个 MG90S 舵机与 HC-05 蓝牙，整机成本约 $60–100。",
        article="文章称项目含完整硬件清单、接线说明和固件，支持基本动作与蓝牙控制，是入门级桌面人形代表。",
        readme="README 副标题「Full featured 21 DOF 3D Printed Humanoid Robot based on ATmega328P chip」，硬件清单列出 Arduino Pro Micro、HC-05、2 个 74HC4017、MPU-6050（标注未使用）、21 个 MG90S、2 个 5A DC-DC，总成本 $60–$100，另有约 $15 螺丝；README 自带「Description coming soon」占位。",
        caveat="文章「完整固件、支持基本动作」未在已读 README 段落逐项核对；仓库最近提交 2017-10-19，已多年无更新。",
        abbr=[
            ("DOF", "Degrees of Freedom", "自由度"),
            ("MG90S", "MG90S micro servo", "9 g 级微型舵机"),
        ],
    ),
    dict(
        n=3,
        slug="zowi",
        name="Zowi",
        repo="JavierIH/zowi",
        price="约 €100",
        last="2026-10-16",
        lic="CC BY-SA 4.0（LICENSE 文件）",
        extra="https://github.com/bq/zowiLibs",
        tags=["biped", "education", "open-source", "arduino", "freecad"],
        one="Zowi 是西班牙 bq 公司推出的小型教育双足机器人；JavierIH/zowi 仓库存放 bq Zowi 的源文件，机械件用开源工具 FreeCAD 设计。",
        article="文章称 Zowi 强调可深度改装，支持跳舞、距离检测、声音交互、LED 表情，社区有丰富固件库，官方库为 bq/zowiLibs。",
        readme="README 称仓库为 bq Zowi 机器人的源文件，用 FreeCAD 设计，任何人可打开、研究、修改并分发；克隆需 `--recursive` 拉取 ArduSnake 子模块；LICENSE 为 CC BY-SA 4.0。bq/zowiLibs 仓库可达（最近提交 2016-02-08）。",
        caveat="文章的功能清单（跳舞、避障、声音、LED）来自文章，README 前段未逐项列出；JavierIH/zowi 仓库在 JavierIH 账号下，README 称其为 bq Zowi 的源文件（bq/zowiLibs 官方库最近提交 2016-02-08）。",
        abbr=[("FreeCAD", "FreeCAD", "开源参数化 3D CAD")],
    ),
    dict(
        n=5,
        slug="kayra",
        name="Kayra",
        repo="assadollahi/kayra",
        price="价格未确认",
        last="2026-09-13",
        lic="BSD-3-Clause（LICENSE 文件）",
        tags=["humanoid", "open-source", "3d-printing", "community", "freecad"],
        one="Kayra 是 Ramin Assadollahi 发起、由社区演进的易改装 3D 打印开源人形，目标是行走接近人类幼儿速度、手臂能举起玩具，并支持 Arduino、ESP32 等多种控制器。",
        article="文章称使用 FreeCAD 设计、低成本、易修改、可步行，含设计文件、打印指南、接线图与仿真环境，强调「社区共同进化」。",
        readme="README 列出目标：行走接近幼儿、手臂能举起至少一个玩具、搭好即可行走/抓取动作、易充电换电、多种控制器（Arduino、ESP-32 等）、可选 PS2/8Bitdo 遥控、可连接主机做机器学习/视觉、舵机可替换、模块化、易打印易装配（理想情况无需焊接）、软件用主机 Python + 控制器 MicroPython。",
        caveat="「仿真环境」「FreeCAD」等为文章表述，已读 README 段落未见；以仓库当前内容为准。",
        abbr=[("MicroPython", "MicroPython", "面向微控制器的 Python 实现")],
    ),
    dict(
        n=6,
        slug="plen2",
        name="PLEN2",
        repo="plenprojectcompany/PLEN2",
        price="套件约 $499",
        last="2018-12-10",
        lic="仓库根目录未见 LICENSE 文件",
        tags=["humanoid", "open-source", "3d-printing", "arduino", "plen"],
        one="PLEN2 是日本 PLEN Project 的可打印开源人形机器人，仓库含固件（arduino/firmware）与 STL 文件，README 自称「The world's first printable open-source humanoid」。",
        article="文章称它是较早的完整可打印开源人形方案之一，提供完整 STL、固件和动作库，社区有换用 MG90S 舵机的改装版。",
        readme="README 的上手流程：下载仓库 → 购买控制板与舵机 → 安装固件（/arduino/firmware/）→ 打印并装配 /stl/ 下 STL → 配置关节零位。",
        caveat="「世界首个可打印开源人形」是 README 自述，未独立核实；文章的「动作库」「MG90S 改装」未在已读段落核对；仓库最近提交 2018-12-10，且未见 LICENSE 文件，复用前需确认许可。",
        abbr=[("STL", "Stereolithography file", "3D 打印常用网格文件格式")],
    ),
    dict(
        n=7,
        slug="nova-desktop-humanoid",
        name="Nova Desktop Humanoid",
        repo=None,
        price="约 $150",
        last="",
        lic="未核实",
        site="https://www.novarobot.co.uk/",
        tags=["humanoid", "desktop-robot", "3d-printing", "companion", "ai-interaction"],
        one="Nova Desktop Humanoid 在文中被描述为 Hayden Woodger（United Robotics）开发的桌面陪伴型人形：3D 打印结构、少量 ST3215 舵机，支持人脸追踪、语音对话、表情动画与记忆（文章说法）。",
        article="文章称设计文件可在 Printables、Creality Cloud 获取，官网 novarobot.co.uk。",
        readme="文章未给 GitHub 仓库；本次未访问官网或 Printables / Creality Cloud，**无法核对任何细节**。",
        caveat="本节点只有文章表述，价格、功能与「开源」程度均未核实；STL 是否带开源许可不明。",
        abbr=[("ST3215", "ST3215 serial bus servo", "文章提到的舵机型号")],
    ),
    dict(
        n=8,
        slug="bambot",
        name="Bambot",
        repo="timqian/bambot",
        price="约 $300",
        last="2025-07-14",
        lic="Apache-2.0（LICENSE 文件）",
        site="https://bambot.org",
        tags=["open-source", "low-cost", "ai-robot", "browser-control"],
        one="Bambot 是低成本开源 AI 机器人项目（官网 bambot.org，README 标语「Play with open-source, low-cost AI robots」），文章称含 3D 打印件、硬件清单与浏览器端控制界面。",
        article="文章称设计结合常见舵机与模块化结构，Apache-2.0 许可，适合快速搭建可交互机器人并尝试 AI 控制。",
        readme="README 仅有标语与演示视频，LICENSE 为 Apache-2.0；其余细节（硬件清单、浏览器控制）在 README 中未展开。",
        caveat="**文章归入「人形机器人」，但 README 未称其为人形**；形态与硬件细节来自文章，未核实。",
        abbr=[],
    ),
    dict(
        n=9,
        slug="rapiro",
        name="Rapiro",
        repo=None,
        price="约 $390",
        last="",
        lic="未核实",
        site="https://www.rapiro.com/",
        extra="https://github.com/hybridgroup/cylon-rapiro",
        tags=["humanoid", "desktop-robot", "arduino", "raspberry-pi", "kit"],
        one="Rapiro 是日本设计师 Shota Ishiwatari 设计的小型人形机器人套件：12 个舵机、Arduino 兼容主板，可加装 Raspberry Pi 扩展视觉与网络（文章说法）。",
        article="文章称 3D 模型公开（Thingiverse 搜索 Rapiro），社区驱动示例见 hybridgroup/cylon-rapiro。",
        readme="官方站点 rapiro.com 与 Thingiverse 模型未访问，**官方硬件是否开源未核实**；社区仓库 hybridgroup/cylon-rapiro 经 git ls-remote 可达，最近提交 2015-07-15（Cylon.js 驱动示例，README 未读）。",
        caveat="本节点主要是文章表述；开源范围与许可均未核实。",
        abbr=[("Cylon.js", "Cylon.js", "JavaScript 机器人框架（仓库名所指）")],
    ),
    dict(
        n=10,
        slug="valera",
        name="Valera",
        repo="goodlancer-org/valera",
        price="约 $400",
        last="2024-05-30",
        lic="MIT（LICENSE 文件）",
        tags=["social-robot", "open-source", "ai-interaction", "desktop-robot"],
        one="Valera 是 GoodLancer 团队维护的开源软硬件社交机器人，目标是做出尽可能便宜、易搭建易使用、带 AI 的社交机器人。",
        article="文章称提供完整 3D 源文件与 STL，软硬件均开放（MIT），强调社交互动与可及性。",
        readme="README 标语「Open source hardware and software social robot」，项目由 GoodLancer 团队管理，文档在 readthedocs，LICENSE 为 MIT。",
        caveat="**文章归入人形机器人清单，README 称其为社交机器人**，未称人形；价格与 STL 细节来自文章，未核实。",
        abbr=[],
    ),
    dict(
        n=11,
        slug="reachy-mini",
        name="Reachy Mini",
        repo="pollen-robotics/reachy_mini",
        price="约 $399",
        last="2026-10-30",
        lic="Apache-2.0（LICENSE 文件）",
        site="https://pollen-robotics.com/reachy-mini/",
        tags=["open-source", "desktop-robot", "pollen-robotics", "huggingface", "python"],
        one="Reachy Mini 是 Pollen Robotics 的开源、富表现力桌面机器人，面向 hacker 与 AI 开发者；README 提供 Wireless、Lite 与 MuJoCo 仿真三种平台入口。",
        article="文章称它由 Pollen 与 Hugging Face 合作，支持 Python、LLM 对话、表情动作与社区应用，Lite 版连接外部电脑，硬件设计与 SDK 均开放。",
        readme="README：Wireless 版为 Raspberry Pi CM4 + 电池 + WiFi 的全自主版；Lite 版为 USB 连接电脑的开发者版；Simulation 在 MuJoCo 中原型开发；LICENSE 为 Apache-2.0。",
        caveat="**文章归入人形机器人清单；README 称「expressive robot」，未称人形**。库内 `painode-169-reachymini` 只是策展索引条目，不是项目节点，故新建本节点。「与 Hugging Face 合作」「硬件设计开放」来自文章，已读 README 段落未逐项核对。",
        abbr=[
            ("SDK", "Software Development Kit", "软件开发包"),
            ("CM4", "Raspberry Pi Compute Module 4", "树莓派计算模块 4"),
        ],
    ),
    dict(
        n=14,
        slug="robotis-mini",
        name="ROBOTIS MINI",
        repo="ROBOTIS-GIT/emanual",
        price="约 $540",
        last="",
        lic="未核实",
        extra="https://github.com/japonophile/darwin",
        tags=["humanoid", "dynamixel", "kit", "robotis", "education"],
        one="ROBOTIS MINI 在文中被描述为基于 DYNAMIXEL XL-320 的紧凑人形套件，官方设计文件（STL/STP）与文档位于 ROBOTIS-GIT/emanual 的 mini 页面。",
        article="文章称提供设计文件、编程工具与详细文档，社区有 ROS 等二次开发，社区示例为 japonophile/darwin。",
        readme="ROBOTIS-GIT/emanual 与 japonophile/darwin 经 git ls-remote 均可达；darwin 仓库 README 为「Darwin Mini」的 CoppeliaSim（V-REP）场景搭建说明，MIT 许可，最近提交 2020-06-20。emanual 为大型文档仓库，本次未克隆、未核对 mini 页面内容。",
        caveat="官方设计文件的开放范围与许可未核实；darwin 仓库是 Darwin Mini 仿真示例，是否等同于 ROBOTIS MINI 的社区示例以文章为准。",
        abbr=[
            ("XL-320", "DYNAMIXEL XL-320", "ROBOTIS 小型舵机"),
            ("STP", "STEP file", "CAD 交换格式"),
        ],
    ),
    dict(
        n=15,
        slug="halley-ambassador-robot",
        name="Halley",
        repo="johnchoi313/Halley-Ambassador-Robot-001",
        price="约 $500–800",
        last="2015-04-09",
        lic="GPL-2.0（LICENSE 文件）",
        tags=["humanoid", "open-source", "social-robot", "animation", "3d-printing"],
        one="Halley（Halley-Ambassador-Robot-001）是约 2.6 英尺高、用于动画研究的开源人形机器人，文章称可用手机作脸、用于人机交互与表情表达。",
        article="文章称提供完整 3D 设计、软件和详细构建教程（GPL），教程见 Instructables。",
        readme="README 仅一句「A 2.6-ft humanoid robot for use in animation research」；LICENSE 为 GNU GPL v2；最近提交 2015-04-09。",
        caveat="「手机作脸」「完整软件与教程」来自文章，README 未展开，未核实；仓库已多年无更新。",
        abbr=[("GPL", "GNU General Public License", "GNU 通用公共许可证")],
    ),
    dict(
        n=17,
        slug="endeavor-ii-humanoid",
        name="Endeavor II",
        repo="ProtoDane/EndeavorHumanoid",
        price="约 $600",
        last="2026-04-02",
        lic="MIT（LICENSE 文件）",
        tags=["humanoid", "open-source", "3d-printing", "robo-one", "esp32"],
        one="Endeavor II 是 ProtoDane 的 50 cm、4 kg 小型 3D 打印人形，受日本 ROBO-ONE 赛事启发，可做敏捷动作、格斗序列与舞蹈，仓库含 STEP、STL 与代码。",
        article="文章称提供组装好的 STEP 模型，持续迭代固件（ESP32 等）与机械结构，项目仍在更新。",
        readme="README 称高 50 cm、重 4 kg，使用商业现货电子件，运费不同整机成本约 $600–$800，仓库含构建与定制所需的 STEP、STL 与代码，并标注「Initial Open Source Release」；LICENSE 为 MIT。",
        caveat="「ESP32」来自文章，已读 README 段落未见；文章写「约 $600」，README 为 $600–$800。",
        abbr=[("ROBO-ONE", "ROBO-ONE", "日本双足机器人格斗/竞技赛事")],
    ),
    dict(
        n=18,
        slug="xlerobot",
        name="XLeRobot",
        repo="Vector-Wangel/XLeRobot",
        price="约 $660",
        last="2026-10-06",
        lic="Apache-2.0（LICENSE 文件）",
        tags=["mobile-manipulation", "open-source", "lerobot", "bimanual", "low-cost"],
        one="XLeRobot 是基于 LeRobot 生态的低成本双臂移动家庭机器人，README 称起步成本 $660，并提供 MuJoCo + 3DGS 网页体验与双轮底盘版本。",
        article="文章称基于 SO-101 机械臂 + 移动底盘，组装时间短、完整开源，适合家用操作、遥操作与 AI 实验。",
        readme="README 标语「Bringing Embodied AI to Everyone - Cheaper Than an iPhone!」，起步 $660；News 含 2026-01-29 浏览器内 MuJoCo + 3DGS 体验、2025-12-02 发布双轮差速底盘版本；LICENSE 为 Apache-2.0。",
        caveat="**文章归入人形机器人清单，XLeRobot 是双臂移动机器人（非人形）**；SO-101 等硬件细节来自文章，未逐项核对。",
        abbr=[
            ("LeRobot", "LeRobot", "Hugging Face 开源机器人学习框架"),
            ("3DGS", "3D Gaussian Splatting", "三维高斯泼溅"),
        ],
    ),
]

CHECKS = [
    (
        "Bambot",
        "归入「人形机器人」清单",
        "README 未称其为人形（仅称低成本 AI 机器人）",
        "按 README 描述，标注归类存疑",
    ),
    ("Valera", "归入「人形机器人」清单", "README 称社交机器人，未称人形", "同上"),
    (
        "Reachy Mini",
        "归入「人形机器人」清单",
        "README 称 expressive robot（桌面机器人）",
        "同上；库内 painode-169 仅是索引条目",
    ),
    (
        "XLeRobot",
        "归入「人形机器人」清单",
        "双臂移动机器人（SO-101 机械臂 + 移动底盘，据文章）",
        "同上",
    ),
    ("Endeavor II", "价格约 $600", "README 为 $600–$800", "采用 README 区间"),
    (
        "PLEN2 / Halley / RoboPrime / Zowi",
        "文章呈现为现成项目",
        "仓库最近提交分别为 2018 / 2015 / 2017 / 2026（zowi 为个人源文件仓库，官方 zowiLibs 为 2016）",
        "标注停更时间",
    ),
    (
        "Nova / Rapiro（官方）/ InMoov 官网",
        "文章给官网或第三方站点",
        "无 GitHub 官方仓库，未访问官网",
        "只记文章表述，标未核实",
    ),
]


def _yl(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items)


def _links(p: dict[str, Any]) -> str:
    out = []
    if p.get("repo"):
        out.append(f"[GitHub](https://github.com/{p['repo']})")
    if p.get("site"):
        out.append(f"[官网/页面]({p['site']})")
    if p.get("extra"):
        out.append(f"[相关链接]({p['extra']})")
    return "，".join(out) or "（文章未给链接）"


def _repo_src(p: dict[str, Any]) -> str:
    link = f"<https://github.com/{p['repo']}>" if p.get("repo") else "（文章未给 GitHub 仓库）"
    extra = f"- **相关链接：** <{p['extra']}>\n" if p.get("extra") else ""
    site = f"- **官网/页面：** <{p['site']}>\n" if p.get("site") else ""
    return f"""# {p["name"]} 来源归档

> 来源归档（repo / site）

- **标题：** {p["name"]}
- **类型：** repo
- **链接：** {link}
{site}{extra}- **入库日期：** {TODAY}
- **线索来源：** [{BLOG}](../blogs/{BLOG})（GitHubStore，文章 2026-11-02）
- **一句话说明：** {p["one"]}
- **最近提交：** {p["last"] or "未核实"}
- **许可：** {p["lic"]}
- **沉淀到 wiki：** [`wiki/entities/{p["slug"]}.md`](../../wiki/entities/{p["slug"]}.md)

## README / 核查摘录（入库日人工阅读）

{p["readme"]}
"""


def _entity(p: dict[str, Any]) -> str:
    rel = [HUM, OSH, *p.get("rel", []), f"../overview/{MAP}"]
    abbr = (
        "\n".join(f"| {a} | {b} | {c} |" for a, b, c in p["abbr"])
        or "| — | — | 本项目无需额外缩写 |"
    )
    rel_lines = "\n".join(f"- [{r.split('/')[-1][:-3]}]({r})" for r in rel)
    code = f"code: https://github.com/{p['repo']}\n" if p.get("repo") else ""
    summary = p["one"].replace('"', "'")
    src_name = p["slug"].replace("-", "_")
    return f"""---
type: entity
project_id: {p["slug"]}
tags:
{_yl(p["tags"])}
status: complete
updated: {TODAY}
{code}related:
{_yl(rel)}
sources:
  - ../../sources/repos/{src_name}.md
  - ../../sources/blogs/{BLOG}
summary: "{summary}"
---

# {p["name"]}

**{p["name"]}**（{_links(p)}）是 [GitHubStore《19 款能自己造的开源人形机器人》]({URL}) 的第 {p["n"]} 项，文章标价 {p["price"]}。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbr}

## 为什么重要

- 属于文章按成本从低到高排列的「能自己造」清单；横向对比见 [{MAP_TITLE}](../overview/{MAP})。
- 价格为文章标价（{p["price"]}），入库日未核实。

## 核心信息

| 项 | 内容 |
|----|------|
| **链接** | {_links(p)} |
| **许可（核查）** | {p["lic"]} |
| **最近提交** | {p["last"] or "未核实"} |

## 文章说法核对

- **文章：** {p["article"]}
- **README / 核查：** {p["readme"]}
- **差异与保留：** {p["caveat"]}

## 关联页面

{rel_lines}

## 参考来源

- [{src_name}.md](../../sources/repos/{src_name}.md)
- [{BLOG}](../../sources/blogs/{BLOG})
- [原文（微信公众号）]({URL})
"""


def _rows() -> list[tuple[int, str, str, str, str, str]]:
    rows = []
    for r in REUSE:
        rows.append(
            (
                r["n"],
                r["name"],
                r["price"],
                f"[{r['node']}](../../wiki/entities/{r['node']}.md)",
                "复用",
                r["note"],
            )
        )
    for p in NEW:
        rows.append(
            (
                p["n"],
                p["name"],
                p["price"],
                f"[{p['slug']}](../../wiki/entities/{p['slug']}.md)",
                "新建",
                "",
            )
        )
    return sorted(rows)


def _blog() -> str:
    body = "\n".join(f"| {n:02d} | {a} | {b} | {c} | {d} | {e} |" for n, a, b, c, d, e in _rows())
    checks = "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in CHECKS)
    return f"""# 19款能自己造的开源人形机器人，附完整开源地址

> 来源归档（blog / 微信公众号）

- **标题：** 19款能自己造的开源人形机器人，附完整开源地址
- **类型：** blog
- **作者：** 小G / GitHubStore（微信公众号；文末链接上一篇「想进入机器人领域吗？学习这18个开源项目就够了」）
- **原始链接：** {URL}
- **发表日期：** 2026-11-02
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 4598 字符
- **一句话说明：** 按成本从低到高列 19 个能自己 3D 打印或组装的开源人形 / 类人形机器人，给出价格、核心配置与开源地址。
- **说明：** 19 个项目 → 19 个唯一节点：**新建 {len(NEW)}、复用 {len(REUSE)}**。价格为文章表述，入库日未核实；GitHub 仓库经 `git ls-remote` 与浅克隆 README 核对，无 GitHub 仓库的项目只记文章表述。

## 项目 → 节点

| # | 项目 | 价格（文章） | 节点 | 状态 | 备注 |
|---|------|-------------|------|------|------|
{body}

## 文章说法与核查的差异

| 项目 | 文章说法 | 核查 | 处理 |
|------|----------|------|------|
{checks}

## 对 wiki 的映射

- 阅读坐标：[{MAP_TITLE}](../../wiki/overview/{MAP})
- 同账号前一篇 18 项清单见 [robot-open-source-wechat-githubstore-18-projects](../../wiki/overview/robot-open-source-wechat-githubstore-18-projects.md)。
"""


def _map() -> str:
    rows = []
    for n, a, b, c, d, e in _rows():
        rows.append(
            f"| {n} | {a} | {b} | {c.replace('../../wiki/entities/', '../entities/')} | {d} |"
        )
    checks = "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in CHECKS)
    first = [f"../entities/{p['slug']}.md" for p in NEW[:3]]
    body = "\n".join(rows)
    return f"""---
type: overview
tags: [overview, humanoid, open-source, hardware, 3d-printing, education, wechat]
status: complete
updated: {TODAY}
related:
{_yl([*first, HUM.replace("../entities/", "../entities/"), OSH, "./robot-open-source-wechat-githubstore-18-projects.md"])}
sources:
  - ../../sources/blogs/{BLOG}
summary: "GitHubStore 清单《19 款能自己造的开源人形机器人》的阅读坐标：按文章标价从 $50 到 $900 排序，新建 {len(NEW)} / 复用 {len(REUSE)} 个独立节点，并标出文章归类、价格与 README 不一致之处。"
---

# {MAP_TITLE}

> **本页定位**：为 [19款能自己造的开源人形机器人]({URL})（2026-11-02）提供阅读坐标；19 个项目均有唯一详情节点。价格为文章表述，未核实。

## 一句话观点

**这份清单把「能自己造」的机器人按成本从几十美元的 Arduino 玩具人形排到近千美元的 InMoov 上半身；其中数个项目（Bambot、Valera、Reachy Mini、XLeRobot）按 README 并非人形。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DOF | Degrees of Freedom | 自由度 |
| BOM | Bill of Materials | 物料清单 |
| STL | Stereolithography file | 3D 打印网格文件 |

## 为什么单独做这张地图

- 库内已有 6 个相关节点分散在各处，新增 {len(NEW)} 个后需要一个统一入口；
- 文章的归类（人形）与若干项目的 README 自述不一致，需要显式标注。

## 项目一览

| # | 项目 | 价格（文章） | 节点 | 状态 |
|---|------|-------------|------|------|
{body}

## 文章说法与核查的差异

| 项目 | 文章说法 | 核查 | 处理 |
|------|----------|------|------|
{checks}

## 关联页面

- [人形机器人总览](../entities/humanoid-robot.md)
- [开源人形硬件对比](../entities/open-source-humanoid-hardware.md)
- [GitHubStore 18 个开源项目（同账号上一篇）](./robot-open-source-wechat-githubstore-18-projects.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
"""


def _patch_reuse() -> None:
    blog = f"../../sources/blogs/{BLOG}"
    mapl = f"../overview/{MAP}"
    for node in sorted({r["node"] for r in REUSE}):
        path = ROOT / "wiki/entities" / f"{node}.md"
        text = path.read_text(encoding="utf-8")
        head, sep, body = text.partition("\n---\n")
        if not sep:
            raise SystemExit(f"frontmatter 缺失: {path}")
        lines = head.split("\n")

        def _append(key: str, item: str) -> None:
            if item in (x.strip().lstrip("- ").strip() for x in lines):
                return
            idx = next((i for i, x in enumerate(lines) if x.startswith(f"{key}:")), None)
            if idx is None:
                lines.extend([f"{key}:", f"  - {item}"])
                return
            if lines[idx].strip() != f"{key}:":
                raise SystemExit(f"{path}: {key} 为行内写法，需手工补")
            j = idx + 1
            indent = "  "
            if j < len(lines) and lines[j].lstrip().startswith("- "):
                indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            while j < len(lines) and lines[j].lstrip().startswith("- "):
                j += 1
            lines.insert(j, f"{indent}- {item}")

        _append("sources", blog)
        _append("related", mapl)
        lines = [f"updated: {TODAY}" if x.startswith("updated:") else x for x in lines]
        path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def main() -> None:
    txt = Path(sys.argv[1]).read_text(encoding="utf-8")
    slugs: set[str] = set()
    for p in NEW:
        if p["slug"] in slugs or (ROOT / "wiki/entities" / f"{p['slug']}.md").exists():
            raise SystemExit(f"slug 重复或节点已存在: {p['slug']}")
        slugs.add(p["slug"])
    for r in REUSE:
        if not (ROOT / "wiki/entities" / f"{r['node']}.md").exists():
            raise SystemExit(f"复用节点不存在: {r['node']}")
    raw = ROOT / "sources/raw" / BLOG
    raw.write_text(
        f"# 19款能自己造的开源人形机器人\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{URL}\n\n{txt}",
        encoding="utf-8",
    )
    (ROOT / "sources/blogs" / BLOG).write_text(_blog(), encoding="utf-8")
    (ROOT / "wiki/overview" / MAP).write_text(_map(), encoding="utf-8")
    for p in NEW:
        (ROOT / "sources/repos" / f"{p['slug'].replace('-', '_')}.md").write_text(
            _repo_src(p), encoding="utf-8"
        )
        (ROOT / "wiki/entities" / f"{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
    _patch_reuse()
    print(f"Wrote 1 blog, 1 map, {len(NEW)} new entities, patched {len(REUSE)} reuse nodes")


if __name__ == "__main__":
    main()
