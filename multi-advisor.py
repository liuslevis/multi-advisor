from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
except ModuleNotFoundError:
    Console = None
    Panel = None
    Prompt = None


OLLAMA_BASE_URL = "http://localhost:11434/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-max"
EXIT_COMMANDS = {"exit", "quit", "退出", "q"}
LIST_COMMANDS = {"list", "ls", "顾问", "人格"}


AGENTS = {
    "drucker": {
        "name": "彼得·德鲁克",
        "title": "商务增长顾问",
        "prompt": "你是彼得·德鲁克。你只关心客户和价值。每次必须先回答两个核心问题：「你的客户是谁？你在为他们创造什么价值？」你会毫不留情砍掉没有真实需求的功能/想法，只在商务增长领域发言。",
    },
    "jobs": {
        "name": "史蒂夫·乔布斯",
        "title": "产品设计顾问",
        "prompt": "你是史蒂夫·乔布斯。你对「够好」零容忍，只追求极致体验。你会问：「这是最好的体验吗？还是只是能用？」你会删除多余内容、简化流程，让产品/界面更有力量和优雅。",
    },
    "hara": {
        "name": "原研哉",
        "title": "系统架构顾问",
        "prompt": "你是原研哉。你信奉「少即是多」。你不问「怎么更好」，而问「这个东西有必要存在吗？」你会审视文件夹、流程、系统，删除60%多余部分，让核心运转更顺畅。",
    },
    "munger": {
        "name": "查理·芒格",
        "title": "投资决策顾问",
        "prompt": "你是查理·芒格。你用多元思维模型（物理、心理、经济学等）拆解问题。你会用至少2-3个不同学科框架分析，避免单一视角，识别「感觉很好但实际是陷阱」的决策。",
    },
    "buffett": {
        "name": "沃伦·巴菲特",
        "title": "商业专注顾问",
        "prompt": "你是沃伦·巴菲特。你最擅长说「No」。你的核心是护城河和能力圈。你会问：「你有什么别人抄不走的优势？」帮用户专注最擅长的事，拒绝跟风趋势，坚持高端窄众。",
    },
    "musk": {
        "name": "埃隆·马斯克",
        "title": "执行速度顾问",
        "prompt": "你是埃隆·马斯克。你只问一个问题：「你为什么还没开始？」你用第一性原理拆解限制，推动立即行动、发MVP、快速迭代。完美是执行的敌人。",
    },
}


HISTORY: list[dict[str, str]] = []
console = Console() if Console else None


@dataclass(frozen=True)
class AdvisorSettings:
    api_key: str
    model: str
    base_url: str


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value[0] in {'"', "'"} and value[-1] == value[0]:
        value = value[1:-1].strip()
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value or None


def _load_dotenv() -> None:
    env_file = _clean_env_value(os.getenv("MULTI_ADVISOR_ENV_FILE")) or ".env"
    path = Path(env_file)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(raw_value)
        if key and value is not None and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str | None = None) -> str | None:
    value = _clean_env_value(os.getenv(name))
    if value is not None:
        return value
    return default


def _looks_like_ollama_url(value: str | None) -> bool:
    if not value:
        return False
    return "localhost:11434" in value or "127.0.0.1:11434" in value


def load_advisor_settings() -> AdvisorSettings:
    _load_dotenv()
    llm_api_key = _env("LLM_API_KEY")
    openai_api_key = _env("OPENAI_API_KEY")
    qwen_api_key = _env("QWEN_API_KEY")
    dashscope_api_key = _env("DASHSCOPE_API_KEY")

    raw_model = _env("LLM_MODEL") or _env("QWEN_MODEL") or _env("QWEN_MODEL_NAME")
    base_url = _env("LLM_BASE_URL") or _env("OPENAI_BASE_URL") or _env("QWEN_BASE_URL")

    explicit_ollama_model = bool(raw_model and raw_model.startswith("ollama/"))
    using_ollama = explicit_ollama_model or _looks_like_ollama_url(base_url)
    if not using_ollama and raw_model is None and not any(
        [llm_api_key, openai_api_key, qwen_api_key, dashscope_api_key]
    ):
        using_ollama = True

    model = raw_model or ("qwen2.5" if using_ollama else DEFAULT_MODEL)
    if explicit_ollama_model:
        model = raw_model.split("/", 1)[1].strip()

    if not base_url:
        if using_ollama:
            base_url = OLLAMA_BASE_URL
        elif qwen_api_key or dashscope_api_key:
            base_url = DEFAULT_BASE_URL
        else:
            base_url = OPENAI_BASE_URL

    api_key = llm_api_key or openai_api_key or qwen_api_key or dashscope_api_key
    if using_ollama:
        api_key = "ollama"

    if not api_key:
        raise RuntimeError(
            "缺少 API Key。请设置 `LLM_API_KEY` / `OPENAI_API_KEY` / "
            "`QWEN_API_KEY` / `DASHSCOPE_API_KEY`，或启动本地 Ollama。"
        )

    return AdvisorSettings(api_key=api_key, model=model, base_url=base_url)


def create_client(settings: AdvisorSettings) -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def parse_targeted_input(user_input: str) -> tuple[str | None, str]:
    stripped = user_input.strip()
    if not stripped.startswith("@"):
        return None, stripped
    command, _, query = stripped.partition(" ")
    return command[1:].lower(), query.strip()


def build_messages(agent_key: str, user_input: str) -> list[dict[str, str]]:
    agent = AGENTS[agent_key]
    messages = [{"role": "system", "content": agent["prompt"]}]
    messages.extend(HISTORY[-8:])
    messages.append({"role": "user", "content": user_input})
    return messages


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = None
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content).strip()


def get_response(client: OpenAI, settings: AdvisorSettings, agent_key: str, user_input: str) -> str:
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=build_messages(agent_key, user_input),
            temperature=0.7,
        )
    except Exception as exc:
        return f"调用出错: {exc}"

    message = response.choices[0].message
    text = extract_text_content(message.content)
    return text or "模型返回了空内容，请重试。"


def print_status(text: str, style: str | None = None) -> None:
    if console:
        if style:
            console.print(f"[{style}]{text}[/{style}]")
        else:
            console.print(text)
        return
    print(text)


def show_panel(body: str, title: str) -> None:
    if console and Panel:
        console.print(Panel(body, title=title, border_style="blue", padding=(1, 2)))
        return
    print(f"\n=== {title} ===")
    print(body)


def ask_user_input() -> str:
    if console and Prompt:
        return Prompt.ask("\n[bold cyan]你的需求或跟进想法[/bold cyan]").strip()
    return input("\n你的需求或跟进想法: ").strip()


def display_responses(responses: dict[str, str]) -> None:
    for key, response in responses.items():
        agent = AGENTS[key]
        title = f"🔹 {agent['name']} - {agent['title']}"
        show_panel(response, title)


def display_welcome(settings: AdvisorSettings) -> None:
    body = (
        "🌟 多顾问 CLI 工具\n"
        "输入需求/想法 → 6 位传奇人物同时给你出主意\n"
        f"当前模型：{settings.model}\n"
        "命令：\n"
        "  • 直接输入想法（所有顾问响应）\n"
        "  • @jobs xxx（只问某一位顾问）\n"
        "  • list（列出所有顾问人格）\n"
        "  • exit / 退出\n"
        "支持持续跟进对话。"
    )
    if console and Panel:
        console.print(Panel.fit(f"[bold cyan]{body}[/bold cyan]", title="欢迎使用巨人肩膀顾问团"))
        return
    print(body)


def display_agents() -> None:
    lines = []
    for key, agent in AGENTS.items():
        lines.append(f"@{key} · {agent['name']} · {agent['title']}")
        lines.append(f"  人格：{agent['prompt']}")
    show_panel("\n".join(lines), "顾问列表")


def update_history(user_input: str, responses: dict[str, str]) -> None:
    HISTORY.append({"role": "user", "content": user_input})
    for key, response in responses.items():
        HISTORY.append({"role": "assistant", "content": f"[{AGENTS[key]['name']}] {response}"})


def main() -> int:
    try:
        settings = load_advisor_settings()
    except RuntimeError as exc:
        print_status(str(exc), style="red")
        return 1

    client = create_client(settings)
    display_welcome(settings)

    try:
        while True:
            user_input = ask_user_input()
            if not user_input:
                print_status("请输入问题或想法。", style="red")
                continue

            if user_input.lower() in EXIT_COMMANDS:
                print_status("再见！你的决策系统已就绪，继续加油！", style="green")
                return 0

            if user_input.lower() in LIST_COMMANDS:
                display_agents()
                continue

            responses: dict[str, str] = {}
            agent_key, query = parse_targeted_input(user_input)
            if agent_key is not None:
                if agent_key not in AGENTS:
                    print_status("没找到这个顾问哦~", style="red")
                    continue
                if not query:
                    print_status("请在顾问名后面补充问题，例如：@jobs 首页该怎么改。", style="red")
                    continue
                print_status(f"正在咨询 {AGENTS[agent_key]['name']}...", style="yellow")
                responses[agent_key] = get_response(client, settings, agent_key, query)
            else:
                print_status("正在召开顾问团会议...", style="yellow")
                for key in AGENTS:
                    print_status(f"  咨询 {AGENTS[key]['name']} ...")
                    responses[key] = get_response(client, settings, key, user_input)

            display_responses(responses)
            update_history(user_input, responses)
    except (EOFError, KeyboardInterrupt):
        print_status("\n再见！你的决策系统已就绪，继续加油！", style="green")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
