"""把 robots.json 里各台机器人的 URDF 与可视网格下载到 assets/robots/<id>/（不入库）。

Usage: python3 fetch_robots.py            # 全部
       python3 fetch_robots.py g1 ur5      # 只取这几台

路径照 Robot Description Gallery Online 的 download.js 解析：package://<pkg>/... 按 packages 映射到仓库内目录，
相对路径按 URDF 所在目录；文件按仓库内路径原样存在 assets/robots/<id>/ 下，舞台用同一套 packages / workingPath 读取。
Collada 文件里引用的贴图也一并下载。走 TLS 代理时设置 SSL_CERT_FILE。
上游只发布 xacro 的（create2、jackal）：xacro:include 引到的文件递归下载（$(find pkg) 按 packages 映射），
网格从所有这些文件里找（网格可能作为宏参数写在 <visual> 外面）；展开在舞台加载时做（robot3d.js）。
照画廊的 build_registry.py：$(optenv NAME 默认值) 取默认值；xacro_stubs 里的包（仓库里没有的 ROS 包，比如 Jackal 的雷达驱动）
不下载，展开时当空文件；<xacro:if> / <xacro:unless> 里的 include 和 xacro 里的网格找不到就跳过（默认配置用不到的附件）。
"""
import json
import posixpath
import re
import sys
import urllib.error
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


OPTENV = re.compile(r"^\$\(optenv\s+\S+\s+([^)\s]+)\)$")
COND_OPEN = re.compile(r"<xacro:(?:if|unless)\b[^>]*[^/]>")
COND_CLOSE = re.compile(r"</xacro:(?:if|unless)>")


def resolve(name, here, packages, stubs=()):
    """URDF / xacro 里写的路径 → 仓库内相对路径（here 为所在文件的目录）；xacro_stubs 里的包返回 None。"""
    fallback = OPTENV.match(name)
    if fallback:
        name = fallback.group(1)
    for prefix in ("package://", "$(find "):
        if name.startswith(prefix):
            pkg, _, rest = name[len(prefix):].partition("/")
            pkg = pkg.rstrip(")")
            if pkg in stubs:
                return None
            if pkg not in packages:
                raise LookupError(f"unknown package {pkg} in {name}")
            return posixpath.normpath(posixpath.join(packages[pkg], rest))
    return posixpath.normpath(posixpath.join(here, name))


def mesh_paths(texts, packages, visual_only, stubs=()):
    """texts：{仓库内路径: 文本}。返回网格在仓库内的相对路径（URDF 只取 <visual> 里的，碰撞体不画，不下载）。
    xacro（visual_only=False）里写成表达式、属于 stub 包或未知包的网格跳过：都是默认配置不带的附件。"""
    out = []
    for rel, text in texts.items():
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        blocks = re.findall(r"<visual\b.*?</visual>", text, re.S) if visual_only else [text]
        for block in blocks:
            for name in re.findall(r'<mesh\b[^>]*\bfilename="([^"]+)"', block):
                try:
                    if "${" in name:
                        raise LookupError(f"mesh path with a xacro expression: {name}")
                    path = resolve(name, posixpath.dirname(rel), packages, stubs)
                except LookupError as e:
                    if visual_only:
                        raise SystemExit(str(e))
                    print(f"  skip {e}", file=sys.stderr)
                    continue
                if path and path not in out:
                    out.append(path)
    return out


def xacro_tree(robot_id, base, root_rel, root_text, packages, stubs):
    """xacro 及它 include 的全部文件：{仓库内路径: 文本}。"""
    texts, queue, optional = {root_rel: root_text}, [root_rel], set()
    while queue:
        rel = queue.pop()
        body = re.sub(r"<!--.*?-->", "", texts[rel], flags=re.S)
        for m in re.finditer(r'<xacro:include\b[^>]*?\bfilename="([^"]+)"', body):
            before = body[:m.start()]
            conditional = rel in optional or len(COND_OPEN.findall(before)) > len(COND_CLOSE.findall(before))
            try:
                inc = resolve(m.group(1), posixpath.dirname(rel), packages, stubs)
                if inc and inc not in texts:
                    texts[inc] = save(robot_id, base, inc).decode("utf-8")
                    queue.append(inc)
                    if conditional:
                        optional.add(inc)
            except (LookupError, urllib.error.HTTPError) as e:
                if not conditional:
                    raise SystemExit(f"{m.group(1)}: {e}")
                print(f"  skip optional include {m.group(1)}: {e}", file=sys.stderr)
    return texts


def main():
    ids = sys.argv[1:] or list(MANIFEST)
    for robot_id in ids:
        m = MANIFEST[robot_id]
        text = save(robot_id, m["base"], m["urdf"]).decode("utf-8")
        stubs = m.get("xacro_stubs", [])
        if m["urdf"].endswith(".xacro"):
            tree = xacro_tree(robot_id, m["base"], m["urdf"], text, m["packages"], stubs)
            meshes = mesh_paths(tree, m["packages"], False, stubs)
        else:
            meshes = mesh_paths({m["urdf"]: text}, m["packages"], True)
        total = 0
        for rel in meshes:
            try:
                data = save(robot_id, m["base"], rel)
            except urllib.error.HTTPError as e:
                if not m["urdf"].endswith(".xacro"):
                    raise
                print(f"  skip mesh {rel}: {e}", file=sys.stderr)   # xacro 里默认配置用不到的附件网格
                continue
            total += len(data)
            if rel.lower().endswith(".dae"):
                for tex in re.findall(r"<init_from>([^<]+\.(?:png|jpe?g|tga))</init_from>", data.decode("utf-8", "ignore"), re.I):
                    total += len(save(robot_id, m["base"], posixpath.normpath(posixpath.join(posixpath.dirname(rel), tex))))
        print(f"{robot_id}: {len(meshes)} meshes, {total / 1e6:.1f} MB -> {OUT / robot_id}", file=sys.stderr)


main()
