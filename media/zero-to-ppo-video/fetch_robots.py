"""把 robots.json 里各台机器人的 URDF 与可视网格下载到 assets/robots/<id>/（不入库）。

Usage: python3 fetch_robots.py            # 全部
       python3 fetch_robots.py g1 ur5      # 只取这几台

路径照 Robot Description Gallery Online 的 download.js 解析：package://<pkg>/... 按 packages 映射到仓库内目录，
相对路径按 URDF 所在目录；文件按仓库内路径原样存在 assets/robots/<id>/ 下，舞台用同一套 packages / workingPath 读取。
Collada 文件里引用的贴图也一并下载。走 TLS 代理时设置 SSL_CERT_FILE。
上游只发布 xacro 的（create2、turtlebot3）：xacro:include 引到的文件递归下载（$(find pkg) 按 packages 映射），
网格从所有这些文件里找（网格可能作为宏参数写在 <visual> 外面）；展开在舞台加载时做（robot3d.js）。
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


def resolve(name, here, packages):
    """URDF / xacro 里写的路径 → 仓库内相对路径（here 为所在文件的目录）。"""
    for prefix in ("package://", "$(find "):
        if name.startswith(prefix):
            pkg, _, rest = name[len(prefix):].partition("/")
            pkg = pkg.rstrip(")")
            if pkg not in packages:
                raise SystemExit(f"unknown package {pkg} in {name}")
            return posixpath.normpath(posixpath.join(packages[pkg], rest))
    return posixpath.normpath(posixpath.join(here, name))


def mesh_paths(texts, packages, visual_only):
    """texts：{仓库内路径: 文本}。返回网格在仓库内的相对路径（URDF 只取 <visual> 里的，碰撞体不画，不下载）。"""
    out = []
    for rel, text in texts.items():
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        blocks = re.findall(r"<visual\b.*?</visual>", text, re.S) if visual_only else [text]
        for block in blocks:
            for name in re.findall(r'<mesh\b[^>]*\bfilename="([^"]+)"', block):
                if "${" in name:
                    raise SystemExit(f"mesh path with a xacro expression: {name}")
                path = resolve(name, posixpath.dirname(rel), packages)
                if path not in out:
                    out.append(path)
    return out


def xacro_tree(robot_id, base, root_rel, root_text, packages):
    """xacro 及它 include 的全部文件：{仓库内路径: 文本}。"""
    texts, queue = {root_rel: root_text}, [root_rel]
    while queue:
        rel = queue.pop()
        body = re.sub(r"<!--.*?-->", "", texts[rel], flags=re.S)
        for name in re.findall(r'<xacro:include\b[^>]*?\bfilename="([^"]+)"', body):
            inc = resolve(name, posixpath.dirname(rel), packages)
            if inc not in texts:
                texts[inc] = save(robot_id, base, inc).decode("utf-8")
                queue.append(inc)
    return texts


def main():
    ids = sys.argv[1:] or list(MANIFEST)
    for robot_id in ids:
        m = MANIFEST[robot_id]
        text = save(robot_id, m["base"], m["urdf"]).decode("utf-8")
        if m["urdf"].endswith(".xacro"):
            meshes = mesh_paths(xacro_tree(robot_id, m["base"], m["urdf"], text, m["packages"]), m["packages"], False)
        else:
            meshes = mesh_paths({m["urdf"]: text}, m["packages"], True)
        total = 0
        for rel in meshes:
            data = save(robot_id, m["base"], rel)
            total += len(data)
            if rel.lower().endswith(".dae"):
                for tex in re.findall(r"<init_from>([^<]+\.(?:png|jpe?g|tga))</init_from>", data.decode("utf-8", "ignore"), re.I):
                    total += len(save(robot_id, m["base"], posixpath.normpath(posixpath.join(posixpath.dirname(rel), tex))))
        print(f"{robot_id}: {len(meshes)} meshes, {total / 1e6:.1f} MB -> {OUT / robot_id}", file=sys.stderr)


main()
