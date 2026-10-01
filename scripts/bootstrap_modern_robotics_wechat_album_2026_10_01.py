#!/usr/bin/env python3
"""Bootstrap ingest: 写个 goodMan · Modern Robotics 原理精读微信专辑（10 篇）."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
JSON_PATH = ROOT / "sources/raw/wechat_modern_robotics_album_4521219024549937157.json"
ALBUM_MD = ROOT / "sources/raw/wechat_modern_robotics_album_4521219024549937157.md"
ALBUM_URL = (
    "https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw=="
    "&action=getalbum&album_id=4521219024549937157"
)

ENTRIES: list[dict] = [
    {
        "blog": "wechat_goodman_cspace_robot_motion_planning.md",
        "ord": 1,
        "wiki": ["configuration-space", "overview/modern-robotics-wechat-principles-series"],
        "one_liner": "从位形、自由度、Grübler 公式到完整/非完整约束，把运动规划的第一张地图画在 C-space 上。",
        "bullets": [
            "位形描述整机关节状态，不是仅末端位置；同末端可对应不同位形。",
            "C-space 维数 = 独立 dof；形状可为环面 $S^1\\times S^1$ 等，与「画成正方形」的展开图区分。",
            "Grübler 公式：$F=6(l-1)-\\sum(6-f_i)$（空间）用于开链/闭链 dof 计数。",
            "完整约束降低 C-space 维数；Pfaffian 非完整约束限制瞬时速度但通常不降维（平面小车典型）。",
            "任务空间 / 工作空间 / C-space 三者的映射与冗余度决定规划在何空间做搜索。",
        ],
    },
    {
        "blog": "wechat_goodman_modern_robotics_ch3_planar_rigid_motion.md",
        "ord": 2,
        "wiki": [
            "lie-group-rigid-body-motions",
            "overview/modern-robotics-wechat-principles-series",
        ],
        "one_liner": "Modern Robotics 第 3 章（1）：向量与参考系、平面 SE(2) 刚体运动与旋转矩阵下标规则。",
        "bullets": [
            "几何向量与坐标列向量分离；同一向量在不同基下坐标不同。",
            "平面刚体位形 $(p_x,p_y,\\theta)$，旋转 $R\\in SO(2)$，$p'=Rp+t$。",
            "下标消去规则：$R_{ac}=R_{ab}R_{bc}$，避免混用同一字母表示不同参考系。",
            "为三维 SO(3)/SE(3)、twist 与齐次矩阵铺垫。",
        ],
    },
    {
        "blog": "wechat_goodman_modern_robotics_ch3_rotation_angular_velocity.md",
        "ord": 3,
        "wiki": ["lie-group-rigid-body-motions", "se3-representation", "unit-quaternion-so3"],
        "one_liner": "第 3 章（2）：SO(3) 旋转矩阵、角速度、指数坐标 $\\omega$ 与矩阵对数。",
        "bullets": [
            "合法旋转矩阵：正交且 $\\det R=1$；姿态与位置分开讨论。",
            "角速度 $\\omega$ 与 $\\dot R = [\\omega]_\\times R$（空间）/ 体坐标形式对照。",
            "有限旋转 = 绕轴 $\\hat\\omega$ 转 $\\theta$；指数坐标 $\\omega=\\hat\\omega\\theta$。",
            "$R=\\exp([\\omega]_\\times)$ 与 $\\log R$ 取回轴角；为 se(3) 指数映射预热。",
        ],
    },
    {
        "blog": "wechat_goodman_homogeneous_transform_why.md",
        "ord": 4,
        "wiki": ["homogeneous-coordinates-transform", "se3-representation"],
        "one_liner": "用 $4\\times4$ 齐次变换把 SE(3) 上的复合刚体运动统一成矩阵乘法。",
        "bullets": [
            "从 $SE(3)=\\{(R,p)\\}$ 到 $T\\in\\mathbb{R}^{4\\times4}$ 的嵌入与分块结构。",
            "点 $(x,y,z,1)$ 与方向 $(x,y,z,0)$ 区分平移敏感/不敏感。",
            "$T_{ac}=T_{ab}T_{bc}$ 链式；Example 3.19 强调用下标管理变换树。",
            "与深蓝专栏齐次坐标文互补，推导对齐 *Modern Robotics* 符号。",
        ],
    },
    {
        "blog": "wechat_goodman_spatial_twist_why.md",
        "ord": 5,
        "wiki": ["spatial-twist-wrench-poe", "lie-group-rigid-body-motions"],
        "one_liner": "运动旋量 $\\mathcal{V}=[\\omega;v]$：统一描述刚体瞬时速度，区分空间 twist 与物体 twist。",
        "bullets": [
            "时变齐次变换 $T(t)$ 导出 $[\\mathcal{V}_s]=\\dot T T^{-1}$ 与 $[\\mathcal{V}_b]=T^{-1}\\dot T$。",
            "$\\omega\\times p + v$ 给出刚体速度场；twist 是其 6 维坐标。",
            "空间/物体 twist 通过 Adjoint 互转；小车例 3.23 对比两种写法。",
            "螺旋轴 $(\\hat\\omega,h)$ 与 pitch 把 twist 与几何运动对应。",
        ],
    },
    {
        "blog": "wechat_goodman_twist_velocity_field_6d.md",
        "ord": 6,
        "wiki": ["spatial-twist-wrench-poe"],
        "one_liner": "从距离不变性推导刚体速度场，说明 twist 是速度场在 6 维李代数上的坐标。",
        "bullets": [
            "刚体上任意两点速度差由 $\\omega\\times$ 决定，形成统一速度场。",
            "选择基点（空间/物体）得到不同 twist 坐标，描述同一物理运动。",
            "为 screw 轴、指数坐标与 PoE 正运动学提供速度侧直觉。",
        ],
    },
    {
        "blog": "wechat_goodman_screw_axis_not_joint_axis.md",
        "ord": 7,
        "wiki": ["spatial-twist-wrench-poe"],
        "one_liner": "关节轴 $\\hat\\omega$ 与螺旋轴 $\\mathcal{S}=(\\hat\\omega,h)$ 不必重合：平移分量来自 $q\\times\\hat\\omega$。",
        "bullets": [
            "绕 $z$ 转 90° 且带平面平移时，瞬时 screw 轴一般不是 $z$。",
            "螺旋运动 = 绕空间某轴匀速转 + 沿轴匀速平移；轴可随位形变。",
            "区分「关节几何轴」与「当前运动螺旋轴」，避免 PoE 建模张冠李戴。",
        ],
    },
    {
        "blog": "wechat_goodman_exponential_coordinates_twist.md",
        "ord": 8,
        "wiki": ["spatial-twist-wrench-poe", "lie-group-rigid-body-motions"],
        "one_liner": "se(3) 指数映射：给定 screw 轴与位移/转角，$T=\\exp([\\mathcal{S}]\\theta)$ 与矩阵对数互逆。",
        "bullets": [
            "$\\mathcal{S}\\theta$ 为 twist 的指数坐标；$\\theta$ 为沿 screw 的广义位移。",
            "平面例 3.26 手算 $\\exp$ / $\\log$ 验证 $T$ 与 $(\\mathcal{S},\\theta)$ 双向转换。",
            "PoE 正运动学把每个关节写成 $\\exp([\\mathcal{S}_i]\\theta_i)$ 的乘积。",
        ],
    },
    {
        "blog": "wechat_goodman_spatial_wrench_why.md",
        "ord": 9,
        "wiki": ["spatial-twist-wrench-poe", "contact-wrench-cone"],
        "one_liner": "力旋量 $\\mathcal{F}=[f;\\tau]$ 把力与力矩合成 6 维量，与 twist 对偶且经 Adjoint 转置变换。",
        "bullets": [
            "同力不同作用点力矩不同；选参考点把 $(f,\\tau)$ 打包成 wrench。",
            "虚功 $\\mathcal{F}^\\top \\mathcal{V}$ 为功率；与 twist 配对做功分析。",
            "wrench 坐标变换用 $\\mathrm{Ad}^T$，与 twist 的 Adjoint 对偶。",
            "六维力传感器读数即 spatial wrench（例 3.28）。",
        ],
    },
    {
        "blog": "wechat_goodman_forward_kinematics_poe.md",
        "ord": 10,
        "wiki": ["forward-kinematics", "spatial-twist-wrench-poe"],
        "one_liner": "正运动学 = 零位形 $M$ 与各关节 $\\exp([\\mathcal{S}_i]\\theta_i)$ 的乘积；空间/物体 PoE 形式对照。",
        "bullets": [
            "平面 3R 与空间 3R 开链：先写零位 $M$，再列各关节 screw $\\mathcal{S}_i$。",
            "空间形式 $T=\\exp([\\mathcal{S}_1]\\theta_1)\\cdots\\exp([\\mathcal{S}_n]\\theta_n)M$。",
            "体坐标形式把指数乘在右侧；与 DH 连乘等价但 screw 来自几何。",
            "常见错误：screw 轴未在零位形下表达、下标系不一致。",
        ],
    },
]


def blog_body(entry: dict, art: dict) -> str:
    raw_rel = art["raw"]
    wiki_links = entry["wiki"]

    def wiki_href(w: str) -> str:
        if w.startswith("overview/"):
            return f"../../wiki/{w}.md"
        if w.startswith("concepts/"):
            return f"../../wiki/{w}.md"
        return f"../../wiki/formalizations/{w}.md"

    mapping = "\n".join(f"- [{w.split('/')[-1]}]({wiki_href(w)})" for w in wiki_links)
    bullets = "\n".join(f"- {b}" for b in entry["bullets"])
    pub = (art.get("publish_time") or "")[:10]
    return f"""# {art["title"]}

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** {art["title"]}
- **类型：** blog
- **作者：** {art.get("author", "写个 goodMan")}（微信公众号）
- **原始链接：** {art["url"]}
- **发表日期：** {pub}
- **入库日期：** {TODAY}
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读]({ALBUM_URL})（第 {entry["ord"]} 篇 / 10）
- **原始抓取落盘：** [`{raw_rel}`](../{raw_rel})
- **一句话说明：** {entry["one_liner"]}

## 核心摘录（归纳，非全文）

{bullets}

## 对 wiki 的映射

{mapping}

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
"""


def write_album_md(data: dict) -> None:
    rows = []
    for art in data["articles"]:
        ent = ENTRIES[art["series_ord"] - 1]
        rows.append(
            f"| {art['series_ord']} | {art['title']} | [链接]({art['url']}) | "
            f"[{ent['blog']}](../blogs/{ent['blog']}) |"
        )
    table = "\n".join(rows)
    ALBUM_MD.write_text(
        f"""# 微信公众号专辑：Modern Robotics 原理精读

- **公众号：** {data["account"]}（`__biz={data["biz"]}`）
- **专辑 ID：** {data["album_id"]}
- **专辑链接：** {ALBUM_URL}
- **入库日期：** {TODAY}
- **抓取方式：** 专辑 HTML 解析 10 条 `data-link`；正文 Camoufox 专辑同会话跳转；清单 JSON 见 [wechat_modern_robotics_album_4521219024549937157.json](./wechat_modern_robotics_album_4521219024549937157.json)

## 专辑目录（10 篇）

| # | 标题 | 文章 URL | 本库 blog 归档 |
|---|------|----------|----------------|
{table}

## 对 wiki 的映射

- 系列父节点：[modern-robotics-wechat-principles-series.md](../../wiki/overview/modern-robotics-wechat-principles-series.md)
- 教材锚点：[modern-robotics-book.md](../../wiki/entities/modern-robotics-book.md)
""",
        encoding="utf-8",
    )


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    by_ord = {a["series_ord"]: a for a in data["articles"]}
    for entry in ENTRIES:
        art = by_ord[entry["ord"]]
        path = ROOT / "sources/blogs" / entry["blog"]
        path.write_text(blog_body(entry, art), encoding="utf-8")
    write_album_md(data)
    print(f"Wrote {len(ENTRIES)} blogs and album md")


if __name__ == "__main__":
    main()
