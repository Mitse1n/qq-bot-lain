# qq-bot

一个基于 QQ 群消息事件的聊天机器人项目（目前集成了 Gemini）。

## 代码结构（便于单元测试）

- `qqbot/bot.py`：核心业务编排（事件 → 入库 → 触发回复 → 分片发送）
- `qqbot/services.py`：对外部系统的集成（事件流、发消息、拉历史、图片处理、Gemini）
- `qqbot/database.py`：SQLite 消息持久化
- 纯函数工具（更容易写单测）：
  - `qqbot/streaming.py`：流式文本分段 `split_message_stream`
  - `qqbot/text_utils.py`：markdown/前缀清理
  - `qqbot/message_parsing.py`：把文本中的 `@12345` 转为 CQ 的 `at` 段

## 运行

容器启动默认执行 `start.sh`，会优先使用 `/app/host/config.yaml`（见 `start.sh`）。

本地运行（示例）：

```bash
python3 qqbot/main.py
```

## 测试

建议用 `uv` 安装开发依赖后再跑：

```bash
uv sync --extra dev
uv run pytest
```

