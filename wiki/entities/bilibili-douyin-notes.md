---
type: entity
tags: [ai-agents, agent-skills, video-notes, speech-recognition, knowledge-management]
status: complete
updated: 2026-10-25
project_id: bilibili-douyin-notes
code: https://github.com/lucky-Luor/bilibili-douyin-notes
summary: "bilibili-douyin-notes 是一个多平台 Agent Skill：先取视频字幕，无字幕时走本地 faster-whisper，再结合抽帧和模板生成课堂 Markdown 笔记。"
related:
  - ./agent-skills-addyosmani.md
  - ../concepts/agentic-coding-software-fundamentals.md
  - ../references/llm-wiki-karpathy.md
sources:
  - ../../sources/repos/bilibili-douyin-notes.md
---

# bilibili-douyin-notes

**bilibili-douyin-notes**（[GitHub](https://github.com/lucky-Luor/bilibili-douyin-notes)，MIT）是遵循 Agent Skills 规范的视频学习工具：从 B 站、抖音、YouTube 或小红书链接提取字幕或音频，必要时在本地转写，并按模板生成可检索的课堂 Markdown 笔记。

## 一句话定义

用“字幕优先、本机 ASR 兜底、按时间戳抽取重点画面”的流程，把长视频整理为结构化笔记，方便复习和纳入个人知识库。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ASR | Automatic Speech Recognition | 无可用字幕时在本机转录音频 |
| PyAV | Python Audio Video | 解码媒体并按时间戳抽帧 |
| Markdown | Markdown markup language | 输出可版本管理、可检索的笔记格式 |
| Agent Skill | Agent Skills specification | 以 `SKILL.md` 声明的代理可调用工作流 |

## 为什么重要

- 输入是公开视频链接，输出是结构化学习材料；不需要先手动下载音频、复制字幕再让代理总结。
- 字幕优先可以减少 ASR 等待与识别错误；无字幕时 `faster-whisper` 在本地运行，项目不依赖付费语音 API。
- 模板包含知识地图、概念卡、自测题和时间索引，产物可回到原视频核对，而非只有一段泛化摘要。

## 核心流程

```mermaid
flowchart LR
  A[视频链接] --> B[平台解析与字幕抓取]
  B --> C{有可用字幕吗}
  C -->|有| D[使用字幕文本]
  C -->|没有| E[本地 faster-whisper 转写]
  D --> F[按时间戳抽取重点画面]
  E --> F
  F --> G[Agent 按课堂模板生成 Markdown]
```

支持 B 站、抖音、YouTube、小红书；YouTube 优先取官方或自动字幕。小红书仅支持视频笔记链接。各平台登录、访问策略和链接形态可能限制抓取。

## 工程实践

1. 将仓库安装到兼容 Agent Skills 的代理技能目录，并按需安装 `requirements.txt` 中相应平台依赖。
2. 先用 README 的 `--probe` 探测链接与分 P 信息；能取到字幕时先用字幕。
3. 无字幕时运行本地 faster-whisper。CPU int8 可运行，但速度受处理器与模型大小影响；GPU 是可选配置。
4. 需要图像辅助时启用 PyAV 截图，并在交付前核对重点帧与笔记时间索引。
5. 将产出的 Markdown 与源视频链接一并归档，保留视频作为事实核对入口。

## 局限与风险

- **开源状态：已开源**，仓库附 MIT 许可证及 `SKILL.md`。本页依据仓库 README 与技能定义归纳，不代表对每个平台抓取的长期可用性保证。
- **外部平台依赖：** 接口、反爬策略、公开视频权限和登录状态变化都可能影响字幕或媒体抓取。
- **ASR 质量：** 专有名词、口音、噪声和低音质会造成错字；课堂笔记中的公式、代码、定义与关键数字应回听核验。
- **内容与隐私：** 下载或缓存的媒体文件可能含个人信息或受版权保护内容，遵守平台条款与当地法律，并按项目说明管理缓存和 Cookie。

## 源码运行时序图

入口：`python scripts/bili_fetch.py "<B站视频链接>" "<工作目录>/bili-notes-<BV号>"`，之后由 `SKILL.md` 指示 Agent 依次调用 `scripts/bili_transcribe.py` 与 `scripts/validate_note.py`。下图只覆盖“单个 B站视频”路径；抖音、YouTube、小红书与整门课（`course_plan.py`）不在图中。

```mermaid
sequenceDiagram
    autonumber
    participant AG as Agent 按 SKILL.md 调度
    participant BF as bili_fetch.py main
    participant BA as B站 API
    participant BT as bili_transcribe.py main
    participant FW as faster-whisper 转写
    participant VN as validate_note.py main

    AG->>BF: python bili_fetch.py 链接 工作目录
    BF->>BF: load_sessdata 读取 SESSDATA extract_bvid
    BF->>BA: x web-interface view 取 aid 标题 分P 时长
    BA-->>BF: 视频信息
    BF->>BF: 写 metadata.json
    loop 每个分P
        BF->>BA: fetch_subtitles 取字幕轨与字幕 URL
        BA-->>BF: 字幕列表 或 空
        alt 字幕可用
            BF->>BF: subtitle_to_text 生成 txt
            alt AI 字幕且 subtitle_relevant 未通过
                BF->>BF: 另存为 存疑 txt 标记 ai_mismatch
            else 其它情况
                BF->>BF: 写 NN 分P txt 标记 cc 或 ai
            end
        else 无字幕
            BF->>BF: 标记 none 或 api_empty
        end
    end
    BF->>BF: 将 subtitle_detail 回写 metadata.json
    BF-->>AG: 末行 JSON 摘要
    AG->>BT: 对缺字幕分P运行 bili_transcribe.py metadata.json
    loop 每个未完成分P txt_done 为假
        BT->>BA: download_audio 下载音频流
        BT->>FW: transcribe 调用 load_model 与设备检测
        FW-->>BT: 转写文本
        BT->>BT: 写 txt 并 mark_done 写 done 标记
        BT-->>AG: 逐集 JSON 回执
    end
    AG->>AG: 依 SKILL.md 第三步按模板写课堂笔记 md
    AG->>VN: python validate_note.py 笔记 --template 与 --meta --txt-dir
    VN-->>AG: JSON 报告 errors 为 0 才算通过
```

图依据：lucky-Luor/bilibili-douyin-notes@1eec86b，实际读过 `SKILL.md`（触发、流程与第三步门禁命令）、`README.md`（前 120 行）、`scripts/bili_fetch.py`（main 全流程与函数清单）、`scripts/bili_transcribe.py`（main 全流程与函数清单）、`scripts/validate_note.py`（main 与 check_note_dispatch 的调用关系）。`fetch_subtitles`、`transcribe`、`load_model` 与 `check_note_dispatch` 内部未逐行读，仅按调用点与 SKILL.md 描述画出；`note_nav.py` 与 `course_plan.py` 未读。

## 关联页面

- [Agent Skills（Addy Osmani）](./agent-skills-addyosmani.md) — 通用编码工作流技能包，可与面向视频学习的 Skill 对照。
- [Agentic Coding 软件工程基础](../concepts/agentic-coding-software-fundamentals.md)
- [Karpathy LLM Wiki](../references/llm-wiki-karpathy.md) — 生成笔记如何接入持续维护的知识库。

## 参考来源

- [项目来源归档](../../sources/repos/bilibili-douyin-notes.md)
- [项目 README](https://github.com/lucky-Luor/bilibili-douyin-notes#readme)
- [Agent Skill 定义](https://github.com/lucky-Luor/bilibili-douyin-notes/blob/main/SKILL.md)
- 源码核对（2026-11-02）：lucky-Luor/bilibili-douyin-notes@1eec86b，读过 SKILL.md、README.md（前 120 行）、scripts/bili_fetch.py、scripts/bili_transcribe.py、scripts/validate_note.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [Agent Skills 规范](https://agentskills.io/) — 技能目录与 `SKILL.md` 的互操作约定。
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 实现的 Whisper 转录工具。
