#!/bin/bash
# scripts/sync_wiki.sh - 自动化 Wiki 维护与统计同步脚本
set -e

# 检查是否提供了描述
if [ -z "$1" ]; then
  echo "❌ 错误: 请提供操作描述 (DESC)"
  echo "用法: ./scripts/sync_wiki.sh \"描述内容\""
  exit 1
fi

DESC=$1

# catalog.md 由 main 上的 export.yml 自动重新生成，PR 不提交（避免合并冲突）
echo "--- 📊 步骤 2: 生成图谱与主页统计 (make graph) ---"
python3 scripts/generate_link_graph.py
python3 scripts/generate_home_stats.py
python3 scripts/graph_exports_sync.py

echo "--- 🚀 步骤 3: 导出全站数据 (make export) ---"
python3 scripts/export_minimal.py

echo "--- 📝 步骤 4: 记录变更日志到 log.d/ 碎片 (make log) ---"
python3 scripts/append_log.py ingest "$DESC"

echo "--- ✅ 同步完成! ---"
echo "派生统计已在本地生成（gitignore，不提交）。运行 'git status' 检查源文件与 log.d/ 碎片并提交。"
