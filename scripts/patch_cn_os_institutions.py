#!/usr/bin/env python3
"""Patch cn-os-* entities with institutions frontmatter + fix overview link."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAPPINGS = REPO / "exports" / "china-opensource-424-mappings.json"
ENTITIES = REPO / "wiki" / "entities"

COMPANY_INST: dict[str, str] = {
    "智元机器人": "agibot",
    "宇树科技": "unitree",
    "北京人形机器人创新中心": "x-humanoid",
    "优必选": "ubtech",
    "傅利叶智能": "fourier",
    "众擎机器人": "engineai",
    "加速进化": "booster",
    "松延动力": "noetix",
    "星动纪元": "roboterax",
    "乐聚机器人": "leju",
    "云深处科技": "deeprobotics",
    "逐际动力": "limx",
    "星海图": "galaxea",
    "星尘智能": "astribot",
    "上海人形机器人创新中心": "openloong",
    "浙江人形机器人创新中心": "zj-humanoid",
    "小米集团": "xiaomi-robotics",
    "小鹏机器人": "xpeng",
    "智身科技": "zsibot",
    "鹿明机器人": "lumos",
    "达妙科技": "dmbots",
    "高擎机电": "hightorque",
    "魔法原子": "magiclab",
    "越疆科技": "dobot",
    "钛虎机器人": "ti5robot",
    "桥介数物": "bridgedp",
    "萝卜派对（RoboParty）": "roboparty",
    "银河通用": "galbot",
    "千寻智能": "spirit-ai",
    "自变量机器人": "x-square-robot",
    "它石智航": "tars-robotics",
    "智在无界": "agilex-ai",
    "智平方": "alphasquare",
    "智澄AI": "zhicheng-ai",
    "生数科技": "shengshu",
    "极佳视界": "gigaai",
    "面壁智能": "modelbest",
    "群核科技": "manycore",
    "阿里巴巴": "alibaba",
    "腾讯机器人实验室": "tencent",
    "字节跳动机器人团队": "bytedance",
    "蚂蚁集团": "ant-group",
    "蚂蚁灵波": "robbyant",
    "地平线": "horizon-robotics",
    "地瓜机器人": "d-robotics",
    "百度智能云": "baidu",
    "大晓机器人": "ace-robotics",
    "大象机器人": "elephant-robotics",
    "灵巧智能": "psibot",
    "灵心巧手": "linkerbot",
    "帕西尼感知科技": "paxini",
    "戴盟机器人": "dexmate",
    "妙动科技": "miaodong",
    "傲意科技": "ohand",
    "光轮智能": "lightwheel",
    "亮源新创": "lightsource",
    "仙工智能": "seer",
    "众为创造": "zhongwei",
    "千觉机器人": "qianjue",
    "原力灵机": "original-intelligence",
    "奥比中光": "orbbec",
    "禾赛科技": "hesai",
    "速腾聚创": "robosense",
    "遨博智能": "aubo",
    "艾利特机器人": "elite-robot",
    "节卡机器人": "jaka",
    "睿尔曼智能": "realman",
    "非夕科技": "flexiv",
    "松灵机器人": "agilex",
    "梅卡曼德": "mech-mind",
    "法奥意威": "fair-innovation",
    "求之科技": "qztech",
    "简智机器人": "jzrobot",
    "玄雅科技": "xuanya",
    "诺亦腾机器人": "noitom",
    "舞肌科技": "wuji-robotics",
}


def patch_entity(path: Path, inst_id: str) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return
    end = text.find("\n---", 3)
    if end == -1:
        return
    fm = text[3:end].strip()
    body = text[end + 4 :]
    if re.search(r"^institutions:\s*\n", fm, re.M):
        fm = re.sub(
            r"^institutions:\s*\n(?:\s+-\s+.+\n)+",
            f"institutions:\n  - {inst_id}\n",
            fm,
            count=1,
        )
    else:
        fm = fm.rstrip() + f"\ninstitutions:\n  - {inst_id}\n"
    if inst_id not in fm and re.search(r"^tags:\s*\[", fm, re.M):
        fm = re.sub(
            r"^(tags:\s*\[)([^\]]*)(\])",
            lambda m: (
                f"{m.group(1)}{m.group(2)}, {inst_id}{m.group(3)}"
                if inst_id not in m.group(2)
                else m.group(0)
            ),
            fm,
            count=1,
            flags=re.M,
        )
    path.write_text(f"---\n{fm}---{body}", encoding="utf-8")


def main() -> None:
    rows = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    patched = 0
    for r in rows:
        slug = r["slug"]
        if not slug.startswith("cn-os-"):
            continue
        p = ENTITIES / f"{slug}.md"
        inst = COMPANY_INST.get(r["company"])
        if inst:
            patch_entity(p, inst)
            patched += 1
    overview = (
        REPO / "wiki/overview/china-domestic-embodied-opensource-76-companies-technology-map.md"
    )
    ot = overview.read_text(encoding="utf-8")
    ot = ot.replace(
        "./hmi-opensource-projects-coverage.md",
        "../queries/hmi-opensource-projects-coverage.md",
    )
    overview.write_text(ot, encoding="utf-8")
    print(f"Patched {patched} entities; fixed overview link")


if __name__ == "__main__":
    main()
