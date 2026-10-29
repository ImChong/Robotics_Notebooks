"""零基础篇讲解视频：episodes/<ep>.py 的旁白 -> 逐句 TTS -> out/<ep>/timeline.json + narration.wav.

Usage: python3 build.py ep001

改自 Robot_Learning_Paper_Notebooks 的 scripts/paper_video/build.py，只换了声线和语速：
论文系列用 zh-CN-YunxiNeural +22%，这个系列用 zh-CN-YunjianNeural +6%，讲得慢一点，也一听就能分出是哪个系列。

Each line plays storyboard time [from, to] of one scene at 1x speed while it is spoken;
if the line runs longer, the scene holds on `to`.
"""
import array
import asyncio
import importlib.util
import json
import os
import random
import re
import ssl
import subprocess
import sys
import wave
from pathlib import Path

import edge_tts
import edge_tts.communicate
import imageio_ffmpeg

# edge-tts pins certifi's bundle; behind a TLS-intercepting proxy point it at SSL_CERT_FILE instead.
if os.environ.get("SSL_CERT_FILE"):
    edge_tts.communicate._SSL_CTX = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])

HERE = Path(__file__).parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "ep001"
_spec = importlib.util.spec_from_file_location(NAME, HERE / "episodes" / f"{NAME}.py")
EPISODE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EPISODE)
SCRIPT, DISPLAY = EPISODE.SCRIPT, EPISODE.DISPLAY
DISPLAY_FN = getattr(EPISODE, "DISPLAY_FN", None)   # 可选：字幕再加工一遍（如把念出来的数写成 $LaTeX$，舞台用 KaTeX 排）
OUT = HERE / "out" / NAME
OUT.mkdir(parents=True, exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VOICE = "zh-CN-YunjianNeural"
RATE = "+6%"
SR = 24000
LEAD = 0.6    # silence before the first line of each scene: the stage fades the scene in during it (stage.html renderAt)
GAP = 0.30    # breath after every line
END_HOLD = 0.5  # 每幕最后一句再多停一会儿：舞台转场淡出要 0.3 s，收尾画面得先看清
# 全片垫一层 −72 dBFS 的底噪，不留整段的数字静音：换幕处有 2 s 多完全为零的静音，有的播放端 / 蓝牙耳机
# 会在长静音后把接下来的第一个字吞掉。这个电平经 AAC 160k 编码后还在，正常音量下听不见
NOISE_DBFS = -72


def display(s):
    for a, b in DISPLAY:
        s = s.replace(a, b)
    return DISPLAY_FN(s) if DISPLAY_FN else s


async def tts(text, mp3, tries=4):
    for k in range(tries):
        try:
            comm = edge_tts.Communicate(text, VOICE, rate=RATE, boundary="SentenceBoundary")
            sents = []
            with open(mp3, "wb") as f:
                async for ch in comm.stream():
                    if ch["type"] == "audio":
                        f.write(ch["data"])
                    elif ch["type"] == "SentenceBoundary":
                        sents.append((ch["offset"] / 1e7, ch["duration"] / 1e7, ch["text"]))
            return sents
        except edge_tts.exceptions.NoAudioReceived:  # the service drops a request now and then
            if k == tries - 1:
                raise
            await asyncio.sleep(2 ** k)


def to_wav(mp3, wav):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1", "-ar", str(SR), str(wav)], check=True)
    with wave.open(str(wav)) as w:
        return w.readframes(w.getnframes())


def chunks(text, start, dur, limit=17):
    """Split one TTS sentence into subtitle lines of <= ~limit chars, timed by char share.
    A $LaTeX$ run (DISPLAY_FN) is never split: it stands in as private-use chars, as many as it shows (\\times = ×, \\circ = °)."""
    maths = []

    def hold(m):
        maths.append(m.group(0))
        shown = re.sub(r"\\[A-Za-z]+|[{}^_ ]", "", m.group(0)[1:-1].replace("\\times", "×").replace("\\circ", "°"))
        return chr(0xE000 + len(maths) - 1) * max(1, len(shown))

    def back(x):
        return re.sub(r"([\ue000-\uf8ff])\1*", lambda m: maths[ord(m.group(1)) - 0xE000], x)

    text = re.sub(r"\$[^$]+\$", hold, text)
    parts = []
    for p in re.split(r"(?<=[，、；：。！？,;:])", text):
        while len(p) > limit + 4:  # no punctuation to break on: cut near the middle, at a space if any
            mid = len(p) // 2
            sp = [k for k in range(mid - 5, mid + 6) if 0 < k < len(p) and p[k] == " "]
            k = min(sp, key=lambda k: abs(k - mid)) if sp else mid
            while 0 < k < len(p) and p[k] == p[k - 1] and "\ue000" <= p[k] <= "\uf8ff":   # 不切在公式中间
                k -= 1
            parts.append(p[:k])
            p = p[k:].lstrip()
        if p.strip():
            parts.append(p)
    lines, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > limit:
            lines.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        lines.append(cur)
    total = sum(len(x) for x in lines) or 1
    out, t = [], start
    for x in lines:
        d = dur * len(x) / total
        out.append({"t0": round(t, 3), "t1": round(t + d, 3), "text": display(back(x).rstrip("，、；：。,;:"))})
        t += d
    return out


async def main():
    segs, subs, pcm = [], [], bytearray()
    t = 0.0
    prev_scene = None
    for i, (scene, a, b, text) in enumerate(SCRIPT):
        mp3, wav = OUT / f"seg{i:02d}.mp3", OUT / f"seg{i:02d}.wav"
        sents = await tts(text, mp3)
        audio = to_wav(mp3, wav)
        alen = len(audio) / 2 / SR
        # 服务偶尔只回一小段音频而不报错：明显短于字数就重合成
        for k in range(3):
            if alen >= len(text) / 25:
                break
            print(f"seg{i:02d} only {alen:.2f}s for {len(text)} chars, retrying")
            await asyncio.sleep(2 ** k)
            sents = await tts(text, mp3)
            audio = to_wav(mp3, wav)
            alen = len(audio) / 2 / SR
        lead = LEAD if scene != prev_scene else 0.0
        last = i == len(SCRIPT) - 1 or SCRIPT[i + 1][0] != scene
        dur = max(b - a, lead + alen + GAP + (END_HOLD if last else 0))
        # alen：这句配音的实际长度。舞台按它把 [from, to] 等比拉伸，画面事件跟着念到的词走（论文系列是 1 倍速播完再停在 to）
        segs.append({"scene": scene, "t0": round(t, 3), "t1": round(t + dur, 3), "from": a, "to": b, "lead": lead, "alen": round(alen, 3)})
        pad = int(round((t + lead) * SR)) - len(pcm) // 2
        pcm += b"\x00\x00" * max(pad, 0) + audio
        for off, d, s in sents:
            # 先换成字幕写法再切行：逐字母念的缩写带空格，按空格切行会把它拆到两行
            subs += chunks(display(s.strip()), t + lead + off, d)
        t += dur
        prev_scene = scene
        print(f"{i:02d} {scene!s:>5} {alen:5.2f}s -> seg {dur:5.2f}s", file=sys.stderr)
    t += 0.8
    pcm += b"\x00\x00" * (int(round(t * SR)) - len(pcm) // 2)
    samples, rng, sd = array.array("h", bytes(pcm)), random.Random(0), 32768 * 10 ** (NOISE_DBFS / 20)
    for k in range(len(samples)):
        samples[k] = max(-32768, min(32767, samples[k] + round(rng.gauss(0, sd))))
    pcm = samples.tobytes()
    with wave.open(str(OUT / "narration.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(pcm))
    json.dump({"total": round(t, 3), "segments": segs, "subs": subs}, open(OUT / "timeline.json", "w"), ensure_ascii=False, indent=1)
    print(f"total {t:.1f}s", file=sys.stderr)


asyncio.run(main())
