# multi-advisor

一个极简 CLI：把同一个问题同时交给 6 位不同风格的顾问。

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env
python multi-advisor.py
```

## 配置

有 API Key 时，设置任意一个即可：

```bash
export OPENAI_API_KEY=your_key
```

也支持：

- `LLM_API_KEY`
- `QWEN_API_KEY`
- `DASHSCOPE_API_KEY`

也可以直接改 `.env`。

如果没配 Key，脚本会默认尝试本地 Ollama。

## 用法

- 直接输入问题：6 位顾问一起回答
- `@jobs 你的问题`：只问某一位顾问
- `list`：查看顾问列表
- `exit`：退出
