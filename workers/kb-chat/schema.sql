-- Robotics Notebooks 只读 AI 问答匿名记录表（不存储访客 IP、User-Agent 或凭证）
CREATE TABLE IF NOT EXISTS chat_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  lang TEXT NOT NULL,
  mode TEXT NOT NULL,
  grounded INTEGER NOT NULL,
  cached INTEGER NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  citation_ids TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at ON chat_logs(created_at DESC);
