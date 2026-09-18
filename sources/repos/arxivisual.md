# arXivisual GitHub 仓库

> 来源归档（ingest）

- **项目名称：** arXivisual
- **GitHub 地址：** <https://github.com/rajshah6/arXivisual>
- **主页：** <https://arxivisual.org>
- **许可证：** 截至 2026-09-18 **根目录无 LICENSE 文件**（GitHub license 字段为 null）
- **核心功能：** arXiv 论文 → 多智能体管线 → Manim 动画 + scrollytelling 前端。
- **入库日期：** 2026-09-18

## 仓库结构（README 对齐）

| 路径 | 作用 |
|------|------|
| `frontend/` | Next.js 16 + React 19 scrollytelling 阅读器 |
| `backend/` | FastAPI + uv；论文 ingest、agent 管线、Manim 渲染 |
| `backend/agents/` | `SectionAnalyzer`、`VisualizationPlanner`、`ManimGenerator` 等 |
| `docs/ARCHITECTURE.md` | 五阶段管线详述 |
| `infra/` | Azure Container Apps Terraform |
| `.github/workflows/` | CI（pytest）、手动 deploy |

## 关键复现路径

1. **Backend：** `cd backend && cp .env.example .env` → `uv sync` → `uv run uvicorn main:app --reload --port 8000`
2. **Frontend：** `cd frontend && npm install && npm run dev`（默认连 `localhost:8000`）
3. **依赖：** Node 20.9+、Python 3.11+、FFmpeg/Cairo/Pango、LaTeX（MathTex）、Azure OpenAI（GPT-5 族部署）
4. **测试：** `cd backend && uv sync --extra dev && uv run pytest tests/`（离线 dummy credentials）

## 技术栈摘要

| 层 | 技术 |
|----|------|
| 前端 | Next.js 16、TanStack Query、Framer Motion、KaTeX |
| 后端 | FastAPI、SQLAlchemy async（PostgreSQL / SQLite） |
| 动画 | Manim Community + manim-voiceover |
| 存储 | Cloudflare R2（生产）/ 本地 `media/videos/` |
| 部署 | Azure Container Apps（非 Vercel 静态导出） |

## 关联 Wiki 页面

- [arXivisual 实体](../../wiki/entities/arxivisual.md)
- [arXiv](../../wiki/entities/arxiv.md)
- [Manim](../../wiki/entities/manim.md)
