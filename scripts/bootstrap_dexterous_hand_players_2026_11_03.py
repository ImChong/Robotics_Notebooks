#!/usr/bin/env python3
"""Bootstrap ingest: Zane Hub《目前灵巧手行业的核心企业，它们的优势体现在哪些方面？》（2026-11-03）。

- 抓取：`scripts/fetch_wechat_article.py`。文章是行业综述，点名 12 家企业 / 产品。
- 8 个新建独立节点（兆威、因时、灵心巧手、智元 SkillHand、傲意、途见、他山、汉威），
  4 个复用（帕西尼、星动纪元、Shadow Hand、特斯拉 Optimus）；综述内容沉淀为一页概念页。
- 企业 / 产品信息全部为作者表述，未核实（未访问官网）。
- 用法：python3 scripts/bootstrap_dexterous_hand_players_2026_11_03.py <抓取的 txt>
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
URL = "https://mp.weixin.qq.com/s/GgZdcerpXldZ0C_KvwFBoA"
BLOG = "wechat_zane_hub_dexterous_hand_core_players_2026-11-03.md"
PAGE = "dexterous-hand-industry-core-players"
PAGE_TITLE = "灵巧手行业核心企业与工程约束（Zane Hub 综述）"

TACT = "../concepts/tactile-sensing.md"
FORCE = "../concepts/force-control-basics.md"
IMP = "../concepts/impedance-control.md"
PAY = "../concepts/dexterous-hand-payload-core-metrics.md"
DEXK = "../concepts/dexterous-kinematics.md"
HW = "../entities/humanoid-robot.md"
SENS = "../overview/humanoid-hardware-101-sensing-end-effectors.md"
SUPPLY = "../overview/humanoid-hardware-101-supply-chain-economics.md"
MASS = "../concepts/humanoid-mass-production-engineering.md"
TRANS = "../overview/humanoid-hardware-101-linear-transmission-bearings.md"

REUSE: list[dict[str, str]] = [
    dict(name="帕西尼 PaXini（DexH13 GEN2）", node="paxini", camp="感知起家"),
    dict(name="星动纪元 Robotera（XHAND1）", node="robotera", camp="整机自研"),
    dict(name="Shadow Robot（Shadow Hand）", node="shadow-hand", camp="海外科研标杆"),
    dict(name="特斯拉 Optimus 手部", node="tesla-optimus", camp="整机自研 / 海外"),
]

NEW: list[dict[str, Any]] = [
    dict(
        slug="zhaowei-zwhand-dexterous-hand",
        name="兆威机电 ZWHAND 灵巧手",
        camp="零部件下探（微型传动）",
        one="兆威机电从微型传动系统（小模数齿轮、微型减速器、精密注塑件）向下延伸做灵巧手，ZWHAND 系列走「驱动单元放进指关节」的短传动链路线。",
        facts=[
            "文章称其底子是微型传动：消费电子、汽车电子、医疗设备领域多年的小模数齿轮、微型减速器与精密注塑，模具、齿轮、注塑、组装垂直整合。",
            "2024 年 11 月发布把驱动单元放进指关节的灵巧手：17 个主动自由度、0.1 秒级响应、单指节精度误差低于 0.3 度、核心组件寿命超过 1 万小时。",
            "2025 年 ZWHAND 品牌推出 LM06、DM17、DM20 三个型号，分别内置 6、17、20 个微型电机；把微型电机、减速器、力觉传感器与控制模块集成在整手结构里。",
            "投资途见科技（可拉伸多模态电子皮肤），意图把「动得准」和「摸得准」绑在一起；苏州基地总投资约 12 亿元、建筑面积超 10 万平方米，具备年产数百万套微型驱动系统的能力。",
        ],
        edge="短传动链响应快、损失小、结构紧凑，代价是手指体积难压缩、散热与维修更棘手；触觉依赖合作。文章称这与特斯拉把执行器移到前臂的长传动链是相反的工程取舍。",
        row="17/20 主动自由度、0.1s 响应、指节误差 <0.3°、核心件寿命 >1 万小时",
        related=[TRANS, MASS, SUPPLY],
    ),
    dict(
        slug="inspire-robots-rh56-dexterous-hand",
        name="因时机器人 RH56 灵巧手",
        camp="零部件下探（伺服电缸）",
        one="因时机器人（成立于 2016 年）把微型伺服电缸做成标准化产品，RH56 系列灵巧手是多根微型电缸加机构件的组合，主打「够用、可靠、买得到」。",
        facts=[
            "RH56 系列：6 个自由度、12 个关节，指尖输出力约 3 kg；FTP 系列增加触觉感知。",
            "模块化：电缸行程、推力、编码器、控制接口有标准可循，整手开发不必从电机绕线开始，成本也更易控制。",
            "文章称其是国内最早把灵巧手价格做到万元级的企业之一，公开报道中单手价格约 5 万元级（此处原文数字表述有出入，见核对）。",
        ],
        edge="6 个自由度能完成抓、握、捏等基础动作，但手内旋转、精细捏取、工具使用等无法与高自由度方案相比；更适合工业分拣、科研教学、服务机器人。",
        row="6 自由度、12 关节、指尖力约 3 kg、万元级（文章另写单手约 5 万元级）",
        related=[PAY, DEXK],
        check="原文先说「万元级」又说「公开报道中单手价格约 5 万元级」，口径不一致，本页不判断实际价格。",
    ),
    dict(
        slug="linkerbot-linker-hand",
        name="灵心巧手 Linker Hand",
        camp="灵巧手专营",
        one="灵心巧手的 Linker Hand 系列覆盖 11 到 42 个自由度，用「自由度矩阵」覆盖长尾需求，并推开源灵巧手换取开发者与场景数据。",
        facts=[
            "Linker Hand 系列 11–42 自由度：工业版 25–30、科研版最高 42，单指最高 9 个自由度、360 度旋转。",
            "T10 与 T20 售价分别为 19999 元与 49999 元，把高自由度灵巧手拉到消费级价格区间。",
            "2026 年的 L20 Lite 采用蜗轮蜗杆与连杆混合传动、10 个主动自由度，强调高自由度、高性能与可负担价格。",
            "文章称其还推开源灵巧手，试图用开源生态换开发者与场景数据。库内已有其 SDK / URDF 相关仓库节点（cn-os-linkerhand-*）。",
        ],
        edge="优势是覆盖广、迭代快、价格激进；风险是 42 自由度科研版只能展示上限，工业客户关心的 25–30 自由度版本的寿命、良率与售后仍待验证。",
        row="11–42 自由度，工业版 25–30，科研版 42，单指最高 9 自由度",
        related=[
            DEXK,
            "../entities/cn-os-linkerhand-python-sdk.md",
            "../entities/cn-os-linkerhand-urdf.md",
        ],
    ),
    dict(
        slug="agibot-skillhand",
        name="智元 SkillHand",
        camp="本体自研",
        one="智元的 SkillHand 有 19 个自由度（12 个主动）、最大负载 30 kg，集成视觉与触觉，用真实任务（螺丝刀、倒水、打麻将等）迭代手的设计。",
        facts=[
            "文章称其 19 自由度（12 主动）、最大负载 30 kg，集成视觉与触觉，能完成使用螺丝刀、倒水、打麻将等动作。",
            "文章归纳的优势是场景数据与整机验证：手在真实任务里迭代，比在实验室堆参数更有效。",
        ],
        edge="以自用为主，外供有限。",
        row="19 自由度（12 主动）、30 kg 负载",
        related=[HW, "../entities/agibot-world-2026.md"],
    ),
    dict(
        slug="oymotion-ohand",
        name="傲意科技 OHand 系列",
        camp="假肢背景",
        one="傲意科技从假肢与康复场景切入灵巧手，OHand 系列整手负载 30 kg、单指负载 10 kg，运动控制响应快。",
        facts=[
            "文章称假肢场景对重量、噪音、续航与安全性要求极高，这些工程经验平移到人形机器人是其差异化。",
            "整手负载 30 kg、单指负载 10 kg；产品形态更偏「有力、可靠」而非「灵巧、通用」。",
            "库内已有 ROHand / OYMotion 相关仓库节点（cn-os-rohand-*）。",
        ],
        edge="精细操作与自由度偏弱。",
        row="整手 30 kg、单指 10 kg",
        related=[
            PAY,
            "../entities/cn-os-rohand-gen2-urdf-ros2.md",
            "../entities/cn-os-rohand-ros2-pkg.md",
        ],
        check="「傲意科技」与 OHand、ROHand 仓库的对应关系按文章与库内已有节点命名推定，未访问官网核实。",
    ),
    dict(
        slug="tujian-electronic-skin",
        name="途见科技 可拉伸电子皮肤",
        camp="感知专精（触觉器件）",
        one="途见科技做可拉伸、多模态的电子皮肤：文章给出 0.01 N/cm² 感知灵敏度、100 Hz 以上采样率、超过 100 万次使用寿命，并带温度感知。",
        facts=[
            "文章称其电子皮肤强调可拉伸、多模态、高灵敏、长寿命，并支持温度感知。",
            "兆威机电投资了途见科技，意图把传动与感知打包。",
        ],
        edge="文章对感知专精类企业的共同判断：难点不是做出敏感单元，而是做出一致性好的阵列并在温度变化、反复受力后保持稳定；触觉赛道正被消费电子、汽车电子、医疗传感器企业「降维」进入，价格战可能很快到来。",
        row="灵敏度 0.01 N/cm²、采样率 ≥100 Hz、寿命 >100 万次",
        related=[TACT, SENS],
    ),
    dict(
        slug="tashan-capacitive-tactile",
        name="他山科技 电容式触觉芯片",
        camp="感知专精（触觉器件）",
        one="他山科技在文中被描述为做电容式触觉芯片与阵列的企业（文章仅一句话）。",
        facts=["文章只在「感知专精派」中以一句话提及：电容式触觉芯片和阵列。"],
        edge="文章没有给出具体指标；本节点只有这一句话，其余信息未核实。",
        row="（文章未给指标）",
        related=[TACT, SENS],
    ),
    dict(
        slug="hanwei-flexible-sensing",
        name="汉威科技 柔性传感",
        camp="感知专精（触觉器件）",
        one="汉威科技在文中被描述为在柔性传感与气体传感上有长期积累的企业（文章仅一句话）。",
        facts=["文章只在「感知专精派」中以一句话提及：柔性传感与气体传感的长期积累。"],
        edge="文章没有给出具体指标；本节点只有这一句话，其余信息未核实。",
        row="（文章未给指标）",
        related=[TACT, SENS],
    ),
]


def _yl(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items)


def _entity(p: dict[str, Any]) -> str:
    rel = [*p["related"], f"../concepts/{PAGE}.md"]
    facts = "\n".join(f"- {f}" for f in p["facts"])
    rel_lines = "\n".join(f"- [{r.split('/')[-1][:-3]}]({r})" for r in rel)
    check = p.get("check") or "未访问企业官网或其他来源，文中数字均为作者表述，未核实。"
    summary = p["one"].replace('"', "'")
    return f"""---
type: entity
project_id: {p["slug"]}
tags:
  - dexterous-hand
  - china
  - industry
  - humanoid-hardware
status: complete
updated: {TODAY}
related:
{_yl(rel)}
sources:
  - ../../sources/blogs/{BLOG}
summary: "{summary}"
---

# {p["name"]}

**{p["name"]}** 属于 [Zane Hub 灵巧手行业综述]({URL}) 的「{p["camp"]}」。本页只整理文章表述；指标与价格未独立核实。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度；文章强调应看「有效自由度」而非参数表自由度 |

## 为什么重要

- 在综述的行业分工里，它代表「{p["camp"]}」这一类；横向对比见 [{PAGE_TITLE}](../concepts/{PAGE}.md)。

## 文章所述要点

{facts}

## 优势与边界（作者表述）

{p["edge"]}

## 核对

- {check}

## 关联页面

{rel_lines}

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
- [原文（微信公众号）]({URL})
"""


def _blog() -> str:
    rows = [(r["name"], r["node"], "复用") for r in REUSE] + [
        (p["name"], p["slug"], "新建") for p in NEW
    ]
    body = "\n".join(f"| {a} | [{b}](../../wiki/entities/{b}.md) | {c} |" for a, b, c in rows)
    return f"""# 目前灵巧手行业的核心企业，它们的优势体现在哪些方面？

> 来源归档（blog / 微信公众号）

- **标题：** 目前灵巧手行业的核心企业，它们的优势体现在哪些方面？
- **类型：** blog
- **作者：** Zane Zhang / Zane Hub（微信公众号，第 381 篇原创；文末「个人观点，仅供参考」）
- **原始链接：** {URL}
- **发表日期：** 2026-11-03
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 8974 字符
- **一句话说明：** 先拆灵巧手五模块（传动、驱动、减速与丝杠、感知、控制），再按「零部件下探 / 本体自研 / 感知专精 / 海外标杆」四类梳理 12 家企业，并归纳四个工程矛盾与选型三标准。
- **说明：** 企业与产品数字均为作者表述，未核实；文末付费星球推广不入库。文中说明了一个丝杠推力简化公式 F = 2π·T·η / P（作者算例：T=0.05 N·m、导程 1 mm、η=0.5 → 约 157 N，经核算一致）。

## 企业 → 节点（12 家，12 个唯一节点）

| 企业 / 产品 | 节点 | 状态 |
|-------------|------|------|
{body}

## 对 wiki 的映射

- 综述页：[{PAGE_TITLE}](../../wiki/concepts/{PAGE}.md)
"""


def _page() -> str:
    rows = [(r["name"], r["node"], "复用", r["camp"]) for r in REUSE] + [
        (p["name"], p["slug"], "新建", p["camp"]) for p in NEW
    ]
    idx = "\n".join(f"| {a} | {d} | [{b}](../entities/{b}.md) | {c} |" for a, b, c, d in rows)
    rel = [f"../entities/{p['slug']}.md" for p in NEW[:3]] + [
        PAY,
        TACT,
        FORCE,
        IMP,
        SENS,
        SUPPLY,
        MASS,
    ]
    return f"""---
type: concept
tags: [dexterous-hand, industry, humanoid-hardware, tactile, supply-chain, wechat]
status: complete
updated: {TODAY}
related:
{_yl(rel)}
sources:
  - ../../sources/blogs/{BLOG}
summary: "依据 Zane Hub 综述梳理灵巧手五模块（传动 / 驱动 / 减速丝杠 / 感知 / 控制）、四类企业阵营与四个工程矛盾；12 家企业各有独立节点（企业数据为作者表述，未核实）。"
---

# {PAGE_TITLE}

> 依据 [《目前灵巧手行业的核心企业，它们的优势体现在哪些方面？》]({URL})（Zane Hub，2026-11-03）整理。**企业与产品的参数、价格、产能均为作者表述，本库未核实**；文章结论是「表只能看位置，不能看胜负」。

## 一句话定义

**灵巧手是电机、减速器、传动件、传感器、驱动电路、控制算法与结构件共同决定的系统，任何一项短板都会把整手性能拉回最低水平；真正值得比较的，是企业把哪一段能力做成了可复制、可交付、可摊薄成本的东西。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度 |
| FOC | Field-Oriented Control | 磁场定向控制（驱动层） |
| η | Transmission efficiency | 传动效率，丝杠推力公式中的符号 |

## 为什么重要

- 文章称灵巧手约占人形整机成本的 14%–18%，其成本曲线决定人形机器人能否从展示品变成生产工具（作者表述）。
- 竞争重心从「谁有五指、谁自由度多」转向「一致性、寿命、成本能否同时控制」。

## 五个模块（作者归纳）

| 模块 | 要点 |
|------|------|
| 传动 | 腱绳（掌心空间大，但预紧 / 蠕变 / 迟滞，需重新标定）、连杆（刚性好，体积难压缩、易干涉）、丝杠（保持力好；滑动丝杠效率低 0.3–0.5，行星滚柱可达 0.85 以上但微型化贵）、直驱（无背隙、力反馈透明，但扭矩密度低）；近两年向「电机 + 减速器 + 丝杠」收敛，原因是平衡了刚度、保持能力、供应链与成本，而非理论最优 |
| 驱动 | 空心杯（低惯量、无齿槽，但功率密度低、堵转易温升、成本高）、无刷（效率高，需减速器与复杂驱动）、步进（成本低、保持好，高速扭矩下降）、微型伺服电缸（模块化、一致性好）；优势不在用了哪种电机，而在吃透并匹配结构与控制 |
| 减速与丝杠 | 微型化的门槛在制造而非设计：模数 0.2–0.3 的齿轮齿厚公差仅几微米，塑料收缩、热处理变形、粉末冶金密度均匀性、装配同轴度与润滑都会累积成回差 |
| 感知 | 压阻 / 电容 / 压电 / 光学 / 磁性；产品多是复合设计加温度补偿；评价要看阵列密度、量程、分辨率、采样率、迟滞、温漂、耐久、标定一致性，而不止「能不能测到 0.01 N」 |
| 控制 | 位置 → 速度 → 力矩 → 阻抗 → 力位混合；同样的电机与传感器，手内操作水平可差一个代际，差在参数辨识、摩擦补偿与触觉闭环（见 [阻抗控制]({"./impedance-control.md"})、[力控基础]({"./force-control-basics.md"})） |

## 四个工程矛盾

1. **自由度与重量**：每增一个主动自由度就要加电机、减速器、轴承、走线与驱动通道，重量与成本近似线性、可靠性超线性变差；应看「有效自由度」。
2. **刚度与柔顺**：高刚度精确但碰撞易损，柔顺安全但带宽受限，靠阻抗 / 力位混合控制兼顾。丝杠推力简化式 F = 2π·T·η / P；作者算例 T=0.05 N·m、导程 1 mm、η=0.5 → 约 157 N。
3. **性能与可制造性**：42 自由度可作展示，产线要考虑装配节拍、来料一致性、测试工位与售后；可制造性是把性能做成可重复的过程能力。
4. **单价与寿命**：6 自由度工业手可做到万元级，17–20 自由度要数万元；整手寿命遵循木桶效应，最先失效的零件决定维护周期。

## 企业阵营与节点（12 家）

| 企业 / 产品 | 阵营 | 节点 | 状态 |
|-------------|------|------|------|
{idx}

### 作者对阵营的判断

- **零部件下探**（兆威、因时）：优势在制造与成本，短板在场景与算法。
- **本体自研**（星动纪元、智元、特斯拉）：优势在整机协同与数据，短板在规模与成本。
- **感知专精**（途见、他山、汉威）：优势在材料 / 工艺 / 标定，短板在系统集成；壁垒在「传感器 + 算法 + 数据接口」组合而非单一敏感材料。
- **海外标杆**：Shadow Hand 把灵巧做到极致也把成本推到极致（文章称价格 100–220 万元）；特斯拉 Optimus 手执行器集中在前臂，目标自由度 22、一度传出收敛到 16，说明方案仍未收敛。
- 未来分工（作者预判）：传动与结构由零部件企业供给、触觉由感知企业供给、整机厂做定义与集成。

## 选型三标准（作者）

1. 看是否有真实批量交付记录，而非发布会与演示视频；
2. 看关键指标的测试条件（寿命、温升、一致性）是否可复现；
3. 看接口是否开放、是否支持二次开发与长期维护。

## 关联页面

- [灵巧手负载核心指标](./dexterous-hand-payload-core-metrics.md)
- [触觉感知](./tactile-sensing.md)
- [力控基础](./force-control-basics.md)
- [阻抗控制](./impedance-control.md)
- [人形硬件 101：感知与末端执行器]({SENS})
- [人形硬件 101：供应链与经济]({SUPPLY})
- [人形量产工程能力](./humanoid-mass-production-engineering.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
- [原文（微信公众号）]({URL})
"""


def _patch(node: str) -> None:
    path = ROOT / "wiki/entities" / f"{node}.md"
    text = path.read_text(encoding="utf-8")
    head, sep, body = text.partition("\n---\n")
    if not sep:
        raise SystemExit(f"frontmatter 缺失: {path}")
    lines = head.split("\n")
    blog = f"../../sources/blogs/{BLOG}"
    page = f"../concepts/{PAGE}.md"

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
    _append("related", page)
    lines = [f"updated: {TODAY}" if x.startswith("updated:") else x for x in lines]
    path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def main() -> None:
    txt = Path(sys.argv[1]).read_text(encoding="utf-8")
    for p in NEW:
        if (ROOT / "wiki/entities" / f"{p['slug']}.md").exists():
            raise SystemExit(f"节点已存在: {p['slug']}")
        for r in p["related"]:
            if not (ROOT / "wiki/entities" / r).resolve().exists():
                raise SystemExit(f"关联页不存在: {r}")
    for r in REUSE:
        if not (ROOT / "wiki/entities" / f"{r['node']}.md").exists():
            raise SystemExit(f"复用节点不存在: {r['node']}")
    (ROOT / "sources/raw" / BLOG).write_text(
        f"# 灵巧手行业核心企业\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{URL}\n\n{txt}",
        encoding="utf-8",
    )
    (ROOT / "sources/blogs" / BLOG).write_text(_blog(), encoding="utf-8")
    (ROOT / "wiki/concepts" / f"{PAGE}.md").write_text(_page(), encoding="utf-8")
    for p in NEW:
        (ROOT / "wiki/entities" / f"{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
    for r in REUSE:
        _patch(r["node"])
    print(f"Wrote blog + concept page + {len(NEW)} new entities, patched {len(REUSE)} reuse nodes")


if __name__ == "__main__":
    main()
