"""把 robots.json 里各台机器人的 URDF 与可视网格下载到 assets/robots/<id>/（不入库）。

Usage: python3 fetch_robots.py            # 全部
       python3 fetch_robots.py g1 ur5      # 只取这几台

路径照 Robot Description Gallery Online 的 download.js 解析：package://<pkg>/... 按 packages 映射到仓库内目录，
相对路径按 URDF 所在目录；文件按仓库内路径原样存在 assets/robots/<id>/ 下，舞台用同一套 packages / workingPath 读取。
Collada 文件里引用的贴图也一并下载。走 TLS 代理时设置 SSL_CERT_FILE。
"""
import json
import posixpath
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "robots.json").read_text())["robots"]
OUT = HERE / "assets" / "robots"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def save(robot_id, base, rel):
    dest = OUT / robot_id / rel
    if dest.exists() and dest.stat().st_size:
        return dest.read_bytes()
    data = get(base + rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data


def mesh_paths(urdf_text, urdf_rel, packages):
    """<visual> 里的网格在仓库内的相对路径（碰撞体不画，不下载）。"""
    urdf_dir = posixpath.dirname(urdf_rel)
    out = []
    for visual in re.findall(r"<visual\b.*?</visual>", urdf_text, re.S):
        for name in re.findall(r'<mesh\b[^>]*\bfilename="([^"]+)"', visual):
            if name.startswith("package://"):
                pkg, _, rest = name[len("package://"):].partition("/")
                if pkg not in packages:
                    raise SystemExit(f"unknown package {pkg} in {name}")
                path = posixpath.join(packages[pkg], rest)
            else:
                path = posixpath.join(urdf_dir, name)
            path = posixpath.normpath(path)
            if path not in out:
                out.append(path)
    return out


def main():
    ids = sys.argv[1:] or list(MANIFEST)
    for robot_id in ids:
        m = MANIFEST[robot_id]
        text = save(robot_id, m["base"], m["urdf"]).decode("utf-8")
        meshes = mesh_paths(text, m["urdf"], m["packages"])
        total = 0
        for rel in meshes:
            data = save(robot_id, m["base"], rel)
            total += len(data)
            if rel.lower().endswith(".dae"):
                for tex in re.findall(r"<init_from>([^<]+\.(?:png|jpe?g|tga))</init_from>", data.decode("utf-8", "ignore"), re.I):
                    total += len(save(robot_id, m["base"], posixpath.normpath(posixpath.join(posixpath.dirname(rel), tex))))
        print(f"{robot_id}: {len(meshes)} meshes, {total / 1e6:.1f} MB -> {OUT / robot_id}", file=sys.stderr)


main()
