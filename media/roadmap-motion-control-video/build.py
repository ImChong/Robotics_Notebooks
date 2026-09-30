#!/usr/bin/env python3
"""把 chapters/*.txt 讲解稿编译成讲解视频：解析 → 配音（edge-tts）→ 渲染（Chromium）→ 合成（ffmpeg）。

用法（环境准备见 README.md）：
  python3 build.py tts        # 只生成/补齐配音
  python3 build.py render     # 只渲染帧
  python3 build.py all [章节id前缀]  # 全流程，输出 out/ 下的 mp4 / srt / 章节表 / 讲解稿
"""
import asyncio
import hashlib
import json
import math
import os
import re
import ssl
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CH_DIR = ROOT / "chapters"
TTS_DIR = ROOT / "cache" / "tts"
FRAME_DIR = ROOT / "cache" / "frames"
OUT = ROOT / "out"
FPS = 25
SR = 24000
VOICE = os.environ.get("VOICE", "zh-CN-YunxiNeural")
RATE = os.environ.get("RATE", "+6%")
GAP_CUE, GAP_STEP, GAP_SLIDE, LEAD = 0.30, 0.55, 0.95, 0.35
SUB_W = 37.0  # 每行字幕宽度（按中文字符计）


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ---------------------------------------------------------------- 解析讲解稿
def attrs(s):
    out = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def macro(line):
    """::chap 编号 | 名称 | 核心问题 | 要点1 · 要点2
    ::svg 文件名 | 额外属性（内联 svg/ 目录下的图）"""
    if line.startswith("::svg"):
        p = [x.strip() for x in line[5:].split("|", 1)]
        body = (ROOT / "svg" / p[0]).read_text(encoding="utf-8")
        return f"<div class='svgbox' {p[1] if len(p) > 1 else ''}>{body}</div>"
    if line.startswith("::chap"):
        p = [x.strip() for x in line[6:].split("|")]
        meta = "".join(f"<span>{m.strip()}</span>" for m in (p[3].split("·") if len(p) > 3 else []) if m.strip())
        return (f"<div class='chap'><div class='no'>{p[0]}</div><div class='nm'>{p[1]}</div>"
                f"<div class='q'>{p[2] if len(p) > 2 else ''}</div><div class='meta'>{meta}</div></div>")
    return line


def parse():
    chapters, slides = [], []
    ch = cur = None
    mode = None
    for f in sorted(CH_DIR.glob("*.txt")):
        for n, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.rstrip()
            if line.startswith("# chapter:"):
                ch = attrs(line[len("# chapter:"):])
                ch["slides"] = []
                chapters.append(ch)
                mode = None
                continue
            if line.startswith("## slide:"):
                cur = attrs(line[len("## slide:"):])
                cur.setdefault("level", ch["level"])
                cur.setdefault("tag", ch.get("tag", ""))
                cur.update(chapter=ch["id"], html=[], cues=[], where=f"{f.name}:{n}")
                ch["slides"].append(cur)
                slides.append(cur)
                mode = "html"
                continue
            if mode == "html":
                if line.strip() == "---":
                    mode = "cues"
                else:
                    if line.strip().startswith("::"):
                        cur["html"].append(macro(line.strip()))
                    else:
                        cur["html"].append(re.sub(r"::svg\s+([\w.-]+\.svg)", lambda m: macro("::svg " + m.group(1)), raw))
                continue
            if mode == "cues":
                if not line.strip() or line.lstrip().startswith("//"):
                    continue
                m = re.match(r"^(\d+)\|\s*(.+)$", line.strip())
                if not m:
                    sys.exit(f"{f.name}:{n} 无法解析的字幕行：{line}")
                cur["cues"].append({"step": int(m.group(1)), "text": m.group(2).strip()})
    for i, s in enumerate(slides):
        s["idx"] = i
        s["id"] = f"{s['chapter']}-{i:03d}"
        s["html"] = "\n".join(s["html"])
        if not s["cues"]:
            sys.exit(f"{s['where']} 幻灯片没有旁白")
    return chapters, slides


# ---------------------------------------------------------------- 发音替换（只影响配音，不影响字幕）
PRON = [
    (r"π0\.5|π₀\.₅|π₀\.5", "派零点五"), (r"π0\.7|π₀\.₇", "派零点七"), (r"π0|π₀", "派零"), (r"π", "派"),
    (r"GR00T", "Groot"), (r"L−1|L-1", "L负一"), (r"SE\(3\)", "SE3"), (r"SO\(3\)", "SO3"), (r"so\(3\)", "so3"),
    (r"[Ss]im2[Rr]eal", "sim to real"), (r"[Ss]im2[Ss]im", "sim to sim"), (r"Real2Sim", "real to sim"),
    (r"(\d)\s*kHz", r"\1千赫兹"), (r"(\d)\s*Hz", r"\1赫兹"), (r"\bHz\b", "赫兹"),
    (r"→", "到"), (r"≈", "约"), (r"×", "乘"), (r"±", "正负"), (r"——", "，"), (r"…", "，"),
    (r"DAgger", "Dagger"), (r"\bViT\b", "V I T"), (r"IsaacLab", "Isaac Lab"), (r"\bTSID\b", "T S I D"),
    (r"\bCoM\b", "质心"), (r"\bDOF\b|\bDoF\b", "自由度"), (r"\bKp\b", "K p"), (r"\bKd\b", "K d"),
    (r"ros2_control", "ros2 control"), (r"PREEMPT_RT", "PREEMPT RT"), (r"EtherCAT", "Ether CAT"),
    (r"TensorRT", "Tensor RT"), (r"legged_gym", "legged gym"), (r"rsl_rl", "rsl rl"), (r"_", " "),
    (r"MoCap", "动捕"), (r"\bSysID\b", "Sys ID"), (r"\bWFM\b", "W F M"), (r"\bDiT\b", "D I T"),
    (r"\bMLP\b", "M L P"), (r"\bQKV\b", "Q K V"), (r"「|」|『|』", ""), (r"\bvs\.?\b", "对比"),
    (r"WholeBodyControl", "Whole Body Control"), (r"AdaLN", "Ada L N"), (r"U-Net", "U Net"), (r"RT-2", "RT 2"),
    (r"LeRobot", "Le Robot"), (r"LAFAN1", "LAFAN 1"), (r"Push-T", "Push T"), (r"x86", "x八六"), (r"iLQR", "i L Q R"),
    (r"ε", "epsilon"), (r"γ", "gamma"), (r"λ", "lambda"), (r"ω", "omega"), (r"θ", "theta"), (r"τ", "tau"),
    (r"δ", "delta"), (r"Δ", "delta"), (r"ξ", "ksi"),
]


def tts_text(t):
    for a, b in PRON:
        t = re.sub(a, b, t)
    return t


def tts_path(text):
    h = hashlib.sha1(f"{VOICE}|{RATE}|{text}".encode()).hexdigest()[:16]
    return TTS_DIR / f"{h}.mp3"


async def _synth(sem, text, path):
    import edge_tts
    import edge_tts.communicate as c
    if os.environ.get("TTS_CA_BUNDLE"):  # 走 TLS 代理时指定证书链
        c._SSL_CTX = ssl.create_default_context(cafile=os.environ["TTS_CA_BUNDLE"])
    async with sem:
        for attempt in range(5):
            try:
                comm = edge_tts.Communicate(text, VOICE, rate=RATE)
                tmp = path.with_suffix(".part")
                with open(tmp, "wb") as fh:
                    async for chunk in comm.stream():
                        if chunk["type"] == "audio":
                            fh.write(chunk["data"])
                if tmp.stat().st_size < 1000:
                    raise RuntimeError("empty audio")
                tmp.rename(path)
                return
            except Exception as e:  # noqa: BLE001 网络抖动重试
                await asyncio.sleep(2 ** attempt)
                err = e
        raise RuntimeError(f"TTS 失败：{text[:30]}… {err}")


def run_tts(slides):
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    todo = {}
    for s in slides:
        for c in s["cues"]:
            c["say"] = tts_text(c["text"])
            p = tts_path(c["say"])
            c["mp3"] = p
            if not p.exists():
                todo[p] = c["say"]
    if todo:
        print(f"TTS：需合成 {len(todo)} 句")

        async def main():
            sem = asyncio.Semaphore(6)
            await asyncio.gather(*[_synth(sem, t, p) for p, t in todo.items()])
        asyncio.run(main())


def load_pcm(p):
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(p), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         check=True, capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    thr = 300.0
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx):
        pad = int(0.04 * SR)
        a = a[max(0, idx[0] - pad): min(len(a), idx[-1] + pad)]
    return a


# ---------------------------------------------------------------- 字幕切分
def w(s):
    return sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in s)


BREAK_AFTER = "，。；：、！？）”』」,;:!?)"


def boundaries(t):
    """jieba 分词边界（字符下标集合），避免把一个词拆到两行；没装 jieba 时退化为只按标点断行。"""
    import logging
    sys.path.insert(0, str(ROOT / "vendor"))
    try:
        import jieba
    except ImportError:
        return set()
    jieba.setLogLevel(logging.WARNING)
    pos, out = 0, set()
    for tok in jieba.cut(t):
        pos += len(tok)
        out.add(pos)
    return out


def two_lines(t):
    if w(t) <= SUB_W:
        return t
    bset = boundaries(t)
    best, best_score = None, 1e9
    for i in range(1, len(t)):
        left, right = t[:i], t[i:]
        if w(left) > SUB_W or w(right) > SUB_W:
            continue
        if right[0] in BREAK_AFTER or t[i - 1] in "“（《「":
            continue
        if re.match(r"[A-Za-z0-9.]", t[i - 1]) and re.match(r"[A-Za-z0-9.]", t[i]):
            continue
        score = abs(w(left) - w(right)) * 0.5
        if t[i - 1] in BREAK_AFTER:
            score -= 12
        elif t[i - 1] == " " or t[i] == " " or t[i] in "“（《「":
            score -= 8
        elif i in bset:
            score -= 4
        else:
            score += 10
        if score < best_score:
            best, best_score = i, score
    if best is None:
        best = len(t) // 2
    return t[:best].rstrip() + "\n" + t[best:].lstrip()


def split_sub(t):
    if w(t) <= 2 * SUB_W - 2:
        return [two_lines(t)]
    parts = re.findall(r"[^，。；：！？,;:!?]+[，。；：！？,;:!?]*", t)
    chunks, cur = [], ""
    for p in parts:
        if cur and w(cur + p) > 2 * SUB_W - 4:
            chunks.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:
        if w(c) > 2 * SUB_W:
            # 极长无标点：硬切
            mid = len(c) // 2
            out += [two_lines(c[:mid]), two_lines(c[mid:])]
        else:
            out.append(two_lines(c))
    return out


def speak_weight(s):
    s = s.replace("\n", "")
    n = 0.0
    for ch in s:
        if ord(ch) > 0x2E80:
            n += 1.0 if ch not in BREAK_AFTER else 0.6
        elif ch.isalnum():
            n += 0.5
        else:
            n += 0.15
    return max(n, 1.0)


# ---------------------------------------------------------------- 时间线
def frame_key(s, step, sub):
    h = hashlib.sha1(json.dumps([s["html"], s["level"], s.get("tag", ""), s.get("title", ""), s.get("src", ""),
                                 s.get("stage", ""), step, sub], ensure_ascii=False).encode()).hexdigest()[:20]
    return f"fr_{h}.png"


def timeline(chapters, slides):
    t = 0  # 以帧为单位
    segs, chaps, subs = [], [], []
    audio_parts = []
    first_slide = {c["id"]: c["slides"][0]["idx"] for c in chapters}
    chap_by_first = {v: k for k, v in first_slide.items()}
    names = {c["id"]: c.get("name", c["id"]) for c in chapters}
    for s in slides:
        if s["idx"] in chap_by_first:
            chaps.append((t, names[chap_by_first[s["idx"]]]))
        slide_start = t
        t += round(LEAD * FPS)
        s["frames"] = []
        for i, c in enumerate(s["cues"]):
            a = load_pcm(c["mp3"])
            nfr = math.ceil(len(a) / SR * FPS)
            audio_parts.append((t, a))
            chunks = split_sub(c["text"])
            wts = [speak_weight(x) for x in chunks]
            tot = sum(wts)
            acc = t
            for j, (ck, wt) in enumerate(zip(chunks, wts)):
                end = t + nfr if j == len(chunks) - 1 else acc + round(nfr * wt / tot)
                fk = frame_key(s, c["step"], ck)
                s["frames"].append({"step": c["step"], "sub": ck, "file": fk})
                segs.append([fk, acc, end])
                subs.append((acc, end, ck))
                acc = end
            t += nfr
            nxt = s["cues"][i + 1] if i + 1 < len(s["cues"]) else None
            gap = GAP_SLIDE if nxt is None else (GAP_STEP if nxt["step"] != c["step"] else GAP_CUE)
            t += round(gap * FPS)
            segs[-1][2] = t  # 画面保持到空隙结束
        # 幻灯片开头的 LEAD 段：用第一帧填充
        first = [x for x in segs if x[1] >= slide_start][0]
        first[1] = slide_start
    # 去重帧
    for s in slides:
        seen, uniq = set(), []
        for f in s["frames"]:
            if f["file"] not in seen:
                seen.add(f["file"])
                uniq.append(f)
        s["frames"] = uniq
    return t, segs, audio_parts, chaps, subs


def ts(frames, sep=","):
    ms = round(frames * 1000 / FPS)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render(slides):
    deck = {"slides": [{k: s[k] for k in ("id", "level", "tag", "title", "src", "html", "frames") if k in s}
                       | {"stage": s.get("stage", "")} for s in slides]}
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "cache" / "deck.json").write_text(json.dumps(deck, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["node", str(ROOT / "render.cjs"), str(ROOT / "cache" / "deck.json"), str(FRAME_DIR)], check=True)


def assemble(total, segs, audio_parts, chaps, subs, name):
    OUT.mkdir(exist_ok=True)
    # 音频
    buf = np.zeros(int(total / FPS * SR) + SR, dtype=np.float32)
    for start, a in audio_parts:
        o = int(round(start / FPS * SR))
        buf[o:o + len(a)] += a
    buf = buf[: int(total / FPS * SR)]
    pcm = np.clip(buf, -32767, 32767).astype(np.int16)
    wav = ROOT / "cache" / f"{name}.wav"
    import wave
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    # 画面
    lst = ROOT / "cache" / f"{name}.ffconcat"
    lines = ["ffconcat version 1.0"]
    for fk, a, b in segs:
        lines.append(f"file '{FRAME_DIR / fk}'")
        lines.append(f"duration {(b - a) / FPS:.4f}")
    lines.append(f"file '{FRAME_DIR / segs[-1][0]}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 章节
    meta = ROOT / "cache" / f"{name}.ffmeta"
    m = [";FFMETADATA1", "title=运动控制 → 物理智能全栈成长路线：逐级讲解"]
    for i, (st, nm) in enumerate(chaps):
        en = chaps[i + 1][0] if i + 1 < len(chaps) else total
        m += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(st * 1000 / FPS)}", f"END={round(en * 1000 / FPS)}", f"title={nm}"]
    meta.write_text("\n".join(m) + "\n", encoding="utf-8")
    mp4 = OUT / f"{name}.mp4"
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(wav),
           "-i", str(meta), "-map", "0:v", "-map", "1:a", "-map_metadata", "2", "-map_chapters", "2",
           "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage", "-crf", "22",
           "-pix_fmt", "yuv420p", "-g", str(FPS * 60), "-c:a", "aac", "-b:a", "48k", "-ac", "1",
           "-movflags", "+faststart", "-shortest", str(mp4)]
    subprocess.run(cmd, check=True)
    # 字幕与章节表
    srt = []
    for i, (a, b, text) in enumerate(subs, 1):
        srt += [str(i), f"{ts(a)} --> {ts(b)}", text, ""]
    (OUT / f"{name}.srt").write_text("\n".join(srt), encoding="utf-8")
    (OUT / f"{name}-chapters.txt").write_text(
        "\n".join(f"{ts(st, '.')[:8]} {nm}" for st, nm in chaps) + "\n", encoding="utf-8")
    print(f"输出：{mp4}  时长 {ts(total, '.')}")


def write_script_md(chapters, name):
    md = ["# 运动控制 → 物理智能全栈成长路线：逐级讲解（旁白稿）", "",
          "> 由 `chapters/*.txt` 自动导出；每段旁白下方注明依据。", ""]
    for c in chapters:
        md += [f"## {c.get('name', c['id'])}", ""]
        for s in c["slides"]:
            if s.get("title"):
                md += [f"### {s['title']}", ""]
            md += ["".join(x["text"] for x in s["cues"]), ""]
            if s.get("src"):
                md += [f"*依据：{s['src']}*", ""]
    (OUT / f"{name}-script.md").write_text("\n".join(md), encoding="utf-8")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = sys.argv[2] if len(sys.argv) > 2 else ""
    chapters, slides = parse()
    if only:
        chapters = [c for c in chapters if c["id"].startswith(only)]
        slides = [s for c in chapters for s in c["slides"]]
    run_tts(slides)
    if cmd == "tts":
        return
    total, segs, audio_parts, chaps, subs = timeline(chapters, slides)
    render(slides)
    if cmd == "render":
        return
    name = "roadmap-motion-control-explained" + (f"-{only}" if only else "")
    OUT.mkdir(exist_ok=True)
    assemble(total, segs, audio_parts, chaps, subs, name)
    write_script_md(chapters, name)


if __name__ == "__main__":
    main()
