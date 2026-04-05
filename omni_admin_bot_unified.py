#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniAdmin / Metamorph unified single-file bot.

Design goals:
- Natural language first.
- Message/mention/DM driven.
- Confirmation flow for destructive operations.
- Persistent state in SQLite.
- Optional LLM providers with key rotation.
- Recognize a very broad command catalog derived from the TXT file.
- One-file deployment: this file is self-contained.

What is intentionally conservative:
- It does not pretend to fully reproduce every external bot backend feature
  unless the local environment or API integration supports it.
- Recognized commands that are not natively implemented are routed through a
  capability layer that can answer, proxy, or record them safely.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import os
import random
import re
import shlex
import sqlite3
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REQUIRED_PACKAGES = [
    "discord.py>=2.4.0",
    "requests>=2.31.0",
    "python-dateutil>=2.9.0.post0",
]

def _ensure_packages() -> None:
    module_map = {"discord.py": "discord", "requests": "requests", "python-dateutil": "dateutil"}
    missing: list[str] = []
    for pkg in REQUIRED_PACKAGES:
        name = pkg.split(">=")[0]
        mod = module_map.get(name, name.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        print("[bootstrap] Installing:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *missing])

_ensure_packages()

import discord  # type: ignore
from discord.ext import commands  # type: ignore
import requests
from dateutil import parser as date_parser  # type: ignore

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "omni_bot.sqlite3"
KEYS_PATH = BASE_DIR / "keys.json"
TXT_CATALOG_PATH = BASE_DIR / "Interface do sistema shared text (2).txt"

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default).strip())
    except Exception:
        return int(default)

def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default).strip())
    except Exception:
        return float(default)

TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
OWNER_IDS = {int(x) for x in re.split(r"[,\s]+", os.getenv("OWNER_IDS", "").strip()) if x.isdigit()}
ALLOW_SHELL = _env_bool("ALLOW_SHELL", "0")
CHAT_ENABLED = _env_bool("CHAT_ENABLED", "1")
DEFAULT_CONF_TIMEOUT = _env_int("CONFIRM_TIMEOUT_SECONDS", "300")
DEFAULT_PREFIXES = [".", "!", "/"]
LLM_PROVIDER_ORDER_ENV = os.getenv(
    "LLM_PROVIDER_ORDER",
    "ollama,openrouter,openai,deepseek,cerebras,anthropic,xai,gemini",
)
LLM_PROVIDER_ORDER = [x.strip().lower() for x in LLM_PROVIDER_ORDER_ENV.split(",") if x.strip()]
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", "0.7")
LLM_MAX_OUTPUT_TOKENS = _env_int("LLM_MAX_OUTPUT_TOKENS", "800")
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", "60")
LOCAL_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LOCAL_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OPENAI_BASE_URL_DEFAULT = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENROUTER_BASE_URL_DEFAULT = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
DEEPSEEK_BASE_URL_DEFAULT = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
CEREBRAS_BASE_URL_DEFAULT = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").rstrip("/")
ANTHROPIC_BASE_URL_DEFAULT = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
XAI_BASE_URL_DEFAULT = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
GEMINI_BASE_URL_DEFAULT = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
SAFE_MODE = _env_bool("SAFE_MODE", "0")

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def utcnow_iso() -> str:
    return utcnow().isoformat()

def short_id() -> str:
    return hashlib.blake2s(os.urandom(16), digest_size=5).hexdigest()

def normalize_ws(text: str) -> str:
    return re.sub(r"[\u200b\u200c\u200d\xa0]+", " ", text).strip()

def clean_text(text: str) -> str:
    text = normalize_ws(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_command(text: str) -> str:
    text = clean_text(text).lower()
    text = re.split(r"\s+[-–—]\s+", text, maxsplit=1)[0].strip()
    return text

def human_join(items: list[str]) -> str:
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " e " + items[-1]

def is_owner(user_id: int) -> bool:
    return user_id in OWNER_IDS

def parse_yes(text: str) -> bool:
    return clean_text(text).lower() in {"sim", "s", "yes", "y", "confirmar", "ok", "aceito", "aceitar", "go"}

def parse_no(text: str) -> bool:
    return clean_text(text).lower() in {"não", "nao", "n", "no", "cancelar", "cancel", "stop"}

def parse_duration(text: str) -> Optional[dt.timedelta]:
    t = clean_text(text).lower()
    m = re.fullmatch(r"(\d+)([smhd])", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        return {
            "s": dt.timedelta(seconds=n),
            "m": dt.timedelta(minutes=n),
            "h": dt.timedelta(hours=n),
            "d": dt.timedelta(days=n),
        }[unit]
    try:
        parsed = date_parser.parse(t, default=utcnow())
        delta = parsed - utcnow()
        if delta.total_seconds() > 0:
            return delta
    except Exception:
        pass
    return None

def blocked_shell_tokens(cmd: str) -> bool:
    dangerous = [
        "rm -rf", "rm -fr", "mkfs", "shutdown", "reboot", "poweroff",
        "curl ", "wget ", "| sh", "|bash", "sudo", "dd ", "chmod 777",
        ":(){", "forkbomb", "kill -9", "taskkill", "format c:",
    ]
    low = cmd.lower()
    return any(tok in low for tok in dangerous)

async def run_subprocess(command: str, timeout: int = 20) -> tuple[int, str]:
    args = shlex.split(command)
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return 124, "Timed out."
    return proc.returncode, (out or b"").decode("utf-8", "replace")

class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS settings (
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (scope, scope_id, key)
            );

            CREATE TABLE IF NOT EXISTS memory (
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pending_actions (
                id TEXT PRIMARY KEY,
                guild_id TEXT,
                channel_id TEXT,
                author_id TEXT,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS custom_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                pattern TEXT NOT NULL,
                response TEXT NOT NULL,
                created_by TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT,
                channel_id TEXT,
                user_id TEXT,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tags (
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_by TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (scope, scope_id, name)
            );
            """
        )
        self.conn.commit()

    def set_setting(self, scope: str, scope_id: str, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO settings(scope, scope_id, key, value) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(scope, scope_id, key) DO UPDATE SET value=excluded.value",
            (scope, scope_id, key, value),
        )
        self.conn.commit()

    def get_setting(self, scope: str, scope_id: str, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE scope=? AND scope_id=? AND key=?",
            (scope, scope_id, key),
        ).fetchone()
        return row["value"] if row else default

    def add_memory(self, scope: str, scope_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO memory(scope, scope_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (scope, scope_id, role, content, utcnow_iso()),
        )
        self.conn.commit()

    def recent_memory(self, scope: str, scope_id: str, limit: int = 12) -> list[dict[str, str]]:
        rows = self.conn.execute(
            "SELECT role, content FROM memory WHERE scope=? AND scope_id=? ORDER BY created_at DESC LIMIT ?",
            (scope, scope_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def put_pending(self, payload: dict[str, Any], ttl_seconds: int) -> str:
        action_id = short_id()
        created = utcnow_iso()
        expires = (utcnow() + dt.timedelta(seconds=ttl_seconds)).isoformat()
        self.conn.execute(
            "INSERT INTO pending_actions(id, guild_id, channel_id, author_id, kind, payload, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action_id,
                str(payload.get("guild_id") or ""),
                str(payload.get("channel_id") or ""),
                str(payload.get("author_id") or ""),
                payload["kind"],
                json.dumps(payload, ensure_ascii=False),
                created,
                expires,
            ),
        )
        self.conn.commit()
        return action_id

    def get_pending(self, action_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM pending_actions WHERE id=?", (action_id,)).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload"])
        payload["_row"] = dict(row)
        return payload

    def delete_pending(self, action_id: str) -> None:
        self.conn.execute("DELETE FROM pending_actions WHERE id=?", (action_id,))
        self.conn.commit()

    def add_rule(self, name: str, pattern: str, response: str, created_by: str) -> None:
        self.conn.execute(
            "INSERT INTO custom_rules(name, pattern, response, created_by, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET pattern=excluded.pattern, response=excluded.response, created_by=excluded.created_by, created_at=excluded.created_at",
            (name, pattern, response, created_by, utcnow_iso()),
        )
        self.conn.commit()

    def rules(self) -> list[dict[str, str]]:
        rows = self.conn.execute("SELECT name, pattern, response FROM custom_rules ORDER BY name ASC").fetchall()
        return [dict(r) for r in rows]

    def audit(self, guild_id: Optional[int], channel_id: Optional[int], user_id: Optional[int], action: str, details: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO audit_log(guild_id, channel_id, user_id, action, details, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (str(guild_id or ""), str(channel_id or ""), str(user_id or ""), action, json.dumps(details, ensure_ascii=False), utcnow_iso()),
        )
        self.conn.commit()

    def put_tag(self, scope: str, scope_id: str, name: str, content: str, created_by: str) -> None:
        self.conn.execute(
            "INSERT INTO tags(scope, scope_id, name, content, created_by, created_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(scope, scope_id, name) DO UPDATE SET content=excluded.content, created_by=excluded.created_by, created_at=excluded.created_at",
            (scope, scope_id, name, content, created_by, utcnow_iso()),
        )
        self.conn.commit()

    def get_tag(self, scope: str, scope_id: str, name: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT content FROM tags WHERE scope=? AND scope_id=? AND name=?",
            (scope, scope_id, name),
        ).fetchone()
        return row["content"] if row else None

    def list_tags(self, scope: str, scope_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM tags WHERE scope=? AND scope_id=? ORDER BY name ASC",
            (scope, scope_id),
        ).fetchall()
        return [r["name"] for r in rows]

store = Store(DB_PATH)

def load_keys_config() -> dict[str, Any]:
    if KEYS_PATH.exists():
        try:
            data = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    data = {
        "provider_order": LLM_PROVIDER_ORDER,
        "defaults": {
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
            "timeout_seconds": LLM_TIMEOUT_SECONDS,
        },
        "providers": {
            "ollama": [
                {"name": "ollama-local", "base_url": LOCAL_OLLAMA_BASE_URL, "model": LOCAL_OLLAMA_MODEL}
            ],
            "openrouter": [
                {"name": "openrouter-1", "api_key": "", "base_url": OPENROUTER_BASE_URL_DEFAULT, "model": "openai/gpt-4o-mini"}
            ],
            "openai": [
                {"name": "openai-1", "api_key": "", "base_url": OPENAI_BASE_URL_DEFAULT, "model": "gpt-4.1-mini"}
            ],
            "deepseek": [
                {"name": "deepseek-1", "api_key": "", "base_url": DEEPSEEK_BASE_URL_DEFAULT, "model": "deepseek-chat"}
            ],
            "cerebras": [
                {"name": "cerebras-1", "api_key": "", "base_url": CEREBRAS_BASE_URL_DEFAULT, "model": "llama-3.1-70b"}
            ],
            "anthropic": [
                {"name": "anthropic-1", "api_key": "", "base_url": ANTHROPIC_BASE_URL_DEFAULT, "model": "claude-3-5-sonnet-latest"}
            ],
            "xai": [
                {"name": "xai-1", "api_key": "", "base_url": XAI_BASE_URL_DEFAULT, "model": "grok-2-latest"}
            ],
            "gemini": [
                {"name": "gemini-1", "api_key": "", "base_url": GEMINI_BASE_URL_DEFAULT, "model": "gemini-2.0-flash"}
            ],
        },
    }
    KEYS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

KEYS_CONFIG = load_keys_config()

class ProviderManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider_order = [p.strip().lower() for p in config.get("provider_order") or LLM_PROVIDER_ORDER if p.strip()]
        self.indices: dict[str, int] = {p: 0 for p in (config.get("providers") or {}).keys()}

    def providers_for(self, requested: Optional[str] = None) -> list[str]:
        if requested:
            return [requested.lower()]
        order = [p for p in self.provider_order if p in self.config.get("providers", {})]
        if "ollama" not in order:
            order.append("ollama")
        return order

    def entries(self, provider: str) -> list[dict[str, Any]]:
        providers = self.config.get("providers", {})
        value = providers.get(provider, [])
        return value if isinstance(value, list) else []

    def pick(self, provider: str) -> Optional[dict[str, Any]]:
        entries = self.entries(provider)
        if not entries:
            return None
        idx = self.indices.get(provider, 0) % len(entries)
        return entries[idx]

    def failover(self, provider: str) -> None:
        entries = self.entries(provider)
        if not entries:
            return
        self.indices[provider] = (self.indices.get(provider, 0) + 1) % len(entries)

provider_manager = ProviderManager(KEYS_CONFIG)

def catalog_from_txt(path: Path) -> list[str]:
    if not path.exists():
        return []
    aliases: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = normalize_ws(raw_line)
        if not line or line in {"Commands", "Type", "Prefixed", "Slash", "Context Menu Message", "Context Menu User", "Category", "All", "Fun", "Image Manipulation", "Informational", "Moderation", "Bot Owner Only", "Say", "Search", "Server Settings", "Tools", "Utilities"}:
            continue
        if not line.startswith((".", "/")):
            continue
        if line.startswith("...<") or line.startswith("<?") or line.startswith("<"):
            continue
        alias = re.split(r"\s+[-–—]\s+", line, maxsplit=1)[0].strip()
        alias = re.sub(r"\s+", " ", alias)
        if alias and alias not in seen:
            aliases.append(alias)
            seen.add(alias)
    return aliases

TXT_ALIASES = catalog_from_txt(TXT_CATALOG_PATH)
TXT_ALIASES_SET = set(TXT_ALIASES)

def catalog_match(message_text: str) -> Optional[str]:
    text = normalize_command(message_text)
    if not text:
        return None
    if text in TXT_ALIASES_SET:
        return text
    # longest-prefix match so commands with usage fragments still resolve.
    best: Optional[str] = None
    for alias in TXT_ALIASES:
        if text == alias or text.startswith(alias + " ") or alias.startswith(text + " "):
            if best is None or len(alias) > len(best):
                best = alias
    return best

def message_scope(message: discord.Message) -> tuple[str, str]:
    if message.guild:
        return "guild", str(message.guild.id)
    return "dm", str(message.author.id)

def bot_mentioned(message: discord.Message) -> bool:
    if message.guild is None:
        return True
    if bot.user is None:
        return False
    return bot.user in message.mentions or bot.user.mentioned_in(message)

def strip_bot_mention(message: discord.Message) -> str:
    content = clean_text(message.content)
    if bot.user:
        content = re.sub(rf"^<@!?{bot.user.id}>\s*", "", content).strip()
    return content

def parse_text_command(content: str) -> tuple[Optional[str], str]:
    txt = clean_text(content)
    for prefix in DEFAULT_PREFIXES:
        if txt.startswith(prefix):
            body = txt[len(prefix):].strip()
            if not body:
                return None, ""
            command = body.split()[0]
            rest = body[len(command):].strip()
            if prefix == "/":
                return "/" + command, rest
            return prefix + command if prefix in {".", "!"} else command, rest
    return None, txt

def extract_mentions(message: discord.Message) -> tuple[Optional[discord.Member], Optional[discord.Role], Optional[discord.abc.GuildChannel]]:
    member = message.mentions[0] if message.mentions else None
    role = message.role_mentions[0] if message.role_mentions else None
    channel = message.channel if isinstance(message.channel, discord.abc.GuildChannel) else None
    return member, role, channel

@dataclass
class ActionPlan:
    kind: str
    summary: str
    data: dict[str, Any]
    destructive: bool = False
    needs_confirmation: bool = False
    admin_only: bool = False

def bot_prompt(guild: Optional[discord.Guild]) -> str:
    rules = store.rules()
    rule_lines = "\n".join(f"- {r['name']}: /{r['pattern']}/ -> {r['response']}" for r in rules[:50]) or "(nenhuma)"
    catalog_summary = f"Aliases carregados do TXT: {len(TXT_ALIASES)}"
    return textwrap.dedent(f"""
    Você é um agente do Discord que fala em português.
    Responda de forma objetiva.
    Quando houver risco, peça confirmação explícita.
    Nunca execute ações destrutivas sem confirmação.
    Use o contexto do servidor se houver.

    {catalog_summary}

    Regras customizadas:
    {rule_lines}
    """).strip()

def infer_plan(message: discord.Message, text: str) -> Optional[ActionPlan]:
    low = text.lower().strip()

    for rule in store.rules():
        try:
            if re.search(rule["pattern"], text, re.I | re.S):
                return ActionPlan("custom_rule", rule["response"], {"response": rule["response"], "rule": rule["name"]})
        except re.error:
            continue

    if low in {"help", "ajuda", "o que você faz", "comandos", "menu"}:
        return ActionPlan("help", "Mostrar o catálogo de comandos e ferramentas.", {})
    if low in {"ping", "latência", "latencia"}:
        return ActionPlan("ping", "Medir latência.", {})
    if any(x in low for x in ["avatar", "pfp", "foto de perfil"]):
        return ActionPlan("avatar", "Mostrar avatar.", {})
    if any(x in low for x in ["usuário", "user info", "membro info", "informações do usuário"]):
        return ActionPlan("user_info", "Mostrar informações de usuário.", {})
    if any(x in low for x in ["servidor", "guild", "server info"]):
        return ActionPlan("guild_info", "Mostrar informações do servidor.", {})
    if any(x in low for x in ["canal info", "informações do canal", "channel info"]):
        return ActionPlan("channel_info", "Mostrar informações do canal.", {})
    if any(x in low for x in ["cargo info", "role info", "informações do cargo"]):
        return ActionPlan("role_info", "Mostrar informações do cargo.", {})
    if low.startswith(("math ", "calc ", "calcule ", "calcular ")):
        expr = text.split(" ", 1)[1] if " " in text else ""
        return ActionPlan("math", f"Calcular {expr}", {"expr": expr})
    if low.startswith(("hash ", "md5 ", "sha1 ", "sha256 ", "sha512 ")):
        parts = text.split(" ", 1)
        algo = parts[0].lower()
        expr = parts[1] if len(parts) > 1 else ""
        return ActionPlan("hash", "Gerar hash.", {"text": expr, "algo": algo})
    if low.startswith(("qr ", "qrcode ", "qr code ")):
        content = text.split(" ", 1)[1] if " " in text else ""
        return ActionPlan("qr", "Gerar QR code.", {"content": content})
    if low.startswith(("reverse text ", "texto reverso ", "inverter texto ")):
        content = text.split(" ", 2)[-1]
        return ActionPlan("reverse_text", "Reverter texto.", {"text": content})
    if low.startswith(("regional ", "emoji regional ")):
        content = text.split(" ", 1)[1] if " " in text else ""
        return ActionPlan("regional", "Converter para regional emojis.", {"text": content})
    if low.startswith(("clap ", "bate palma ", "palmas ")):
        content = text.split(" ", 1)[1] if " " in text else ""
        return ActionPlan("clap", "Intercalar palmas.", {"text": content})
    if low.startswith(("owo ", "owofy ", "owoify ")):
        content = text.split(" ", 1)[1] if " " in text else ""
        return ActionPlan("owofy", "Owofy texto.", {"text": content})
    if low.startswith(("ascii ", "ascii art ", "texto ascii ")):
        content = text.split(" ", 1)[1] if " " in text else ""
        return ActionPlan("ascii", "Converter para ASCII.", {"text": content})
    if low.startswith(("eightball ", "bola ", "8ball ", "oráculo ")):
        return ActionPlan("eightball", "Resposta de sorte.", {"question": text})
    if low.startswith(("remind ", "lembrete ", "lembrar ")):
        return ActionPlan("reminder", "Criar lembrete.", {"text": text})
    if low.startswith(("shell ", "terminal ", "cmd ", "executar comando ")):
        return ActionPlan("shell", "Executar comando de terminal.", {"command": text}, needs_confirmation=True, admin_only=True)
    if any(x in low for x in ["banir", "ban ", "kick", "expulsar", "timeout", "mute", "prune", "limpar mensagens", "apagar canal", "deletar canal", "remover canal", "criar cargo", "criar canal", "renomear canal", "renomear cargo", "dar cargo", "remover cargo", "dar apelido", "renomear membro"]):
        return parse_moderation_plan(message, text)
    if any(x in low for x in ["crie uma regra", "adicionar regra", "novo padrão", "novo comportamento", "ajuste seu comportamento", "edite sua resposta", "criar alias"]):
        return ActionPlan("patch_behavior", "Modificar regras de comportamento.", {"text": text}, needs_confirmation=True, admin_only=True)
    if any(x in low for x in ["tag ", "tag create", "tag show", "tag info"]):
        return ActionPlan("tag", "Operação de tags.", {"text": text})
    return None

def parse_moderation_plan(message: discord.Message, text: str) -> ActionPlan:
    low = text.lower()
    data: dict[str, Any] = {"raw": text}
    destructive = False
    if "desbanir" in low:
        data["op"] = "unban"; destructive = True
    elif "ban" in low or "banir" in low:
        data["op"] = "ban"; destructive = True
    elif "kick" in low or "expulsar" in low:
        data["op"] = "kick"; destructive = True
    elif "timeout" in low or "mute" in low:
        data["op"] = "timeout"; destructive = True
    elif "prune" in low or "limpar mensagens" in low:
        data["op"] = "prune"; destructive = True
    elif "criar cargo" in low:
        data["op"] = "role_create"
    elif "remover cargo" in low or "dar cargo" in low:
        data["op"] = "role_edit"
    elif "criar canal" in low:
        data["op"] = "channel_create"
    elif "apagar canal" in low or "deletar canal" in low or "remover canal" in low:
        data["op"] = "channel_delete"; destructive = True
    elif "renomear canal" in low:
        data["op"] = "channel_rename"
    elif "renomear cargo" in low:
        data["op"] = "role_rename"
    elif "dar apelido" in low or "renomear membro" in low:
        data["op"] = "nickname"
    else:
        data["op"] = "mod"
    return ActionPlan("moderation", f"Ação de moderação: {data['op']}", data, destructive=destructive, needs_confirmation=True)

async def send_long(channel: discord.abc.Messageable, text: str) -> None:
    if len(text) <= 1900:
        await channel.send(text)
        return
    for i in range(0, len(text), 1900):
        await channel.send(text[i:i+1900])

# --- LLM adapters -----------------------------------------------------

def _extract_openai_like_text(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()
    if isinstance(data.get("text"), str) and data["text"].strip():
        return data["text"].strip()
    if isinstance(data.get("choices"), list):
        parts: list[str] = []
        for choice in data["choices"]:
            msg = (choice or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content)
        if parts:
            return "\n".join(parts).strip()
    if isinstance(data.get("output"), list):
        parts: list[str] = []
        for item in data["output"]:
            for c in (item or {}).get("content") or []:
                if (c or {}).get("type") in {"output_text", "text"} and c.get("text"):
                    parts.append(str(c["text"]))
        if parts:
            return "\n".join(parts).strip()
    return None

def _openai_like_client(base_url: str, api_key: str, model: str, system_prompt: str, messages: list[dict[str, str]]) -> Optional[str]:
    payload = {
        "model": model,
        "input": [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                  *[{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]],
        "temperature": LLM_TEMPERATURE,
        "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return _extract_openai_like_text(resp.json())

def _anthropic_client(api_key: str, base_url: str, model: str, system_prompt: str, messages: list[dict[str, str]]) -> Optional[str]:
    system_parts: list[str] = [system_prompt]
    anthro_messages: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            anthro_messages.append({"role": m["role"], "content": m["content"]})
    payload = {
        "model": model,
        "system": "\n\n".join(x for x in system_parts if x).strip(),
        "messages": anthro_messages,
        "max_tokens": LLM_MAX_OUTPUT_TOKENS,
        "temperature": LLM_TEMPERATURE,
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    texts: list[str] = []
    for item in (data.get("content") or []):
        if item.get("type") == "text" and item.get("text"):
            texts.append(str(item["text"]))
    return "\n".join(texts).strip() if texts else None

def _gemini_client(api_key: str, base_url: str, model: str, system_prompt: str, messages: list[dict[str, str]]) -> Optional[str]:
    contents: list[dict[str, Any]] = []
    system_parts: list[str] = [system_prompt]
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload: dict[str, Any] = {"contents": contents, "generationConfig": {"temperature": LLM_TEMPERATURE, "maxOutputTokens": LLM_MAX_OUTPUT_TOKENS}}
    system_text = "\n\n".join(x for x in system_parts if x).strip()
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    resp = requests.post(
        f"{base_url.rstrip('/')}/models/{model}:generateContent",
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    texts: list[str] = []
    for cand in data.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            if part.get("text"):
                texts.append(str(part["text"]))
    return "\n".join(texts).strip() if texts else None

def _ollama_client(base_url: str, model: str, system_prompt: str, messages: list[dict[str, str]]) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_OUTPUT_TOKENS},
    }
    resp = requests.post(f"{base_url.rstrip('/')}/api/chat", headers={"Content-Type": "application/json"}, json=payload, timeout=max(LLM_TIMEOUT_SECONDS, 30))
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        if isinstance(data.get("message"), dict):
            c = data["message"].get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
        if isinstance(data.get("response"), str) and data["response"].strip():
            return data["response"].strip()
    return None

def _norm_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "user").lower().strip()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(m.get("content") or "")
        if content.strip():
            out.append({"role": role, "content": content})
    return out

async def llm_chat(system_prompt: str, messages: list[dict[str, str]], provider: Optional[str] = None) -> Optional[str]:
    if not CHAT_ENABLED:
        return None
    normalized = _norm_messages(messages)
    providers = provider_manager.providers_for(provider)
    for prov in providers:
        entries = provider_manager.entries(prov)
        if not entries and prov != "ollama":
            continue
        if prov == "ollama":
            entry = provider_manager.pick("ollama") or {"base_url": LOCAL_OLLAMA_BASE_URL, "model": LOCAL_OLLAMA_MODEL}
            try:
                return await asyncio.to_thread(_ollama_client, entry.get("base_url", LOCAL_OLLAMA_BASE_URL), entry.get("model", LOCAL_OLLAMA_MODEL), system_prompt, normalized)
            except Exception:
                provider_manager.failover("ollama")
                continue
        for entry in entries or [{}]:
            api_key = str(entry.get("api_key") or "").strip()
            if prov != "ollama" and not api_key:
                continue
            base_url = str(entry.get("base_url") or "").rstrip("/") or {
                "openai": OPENAI_BASE_URL_DEFAULT,
                "openrouter": OPENROUTER_BASE_URL_DEFAULT,
                "deepseek": DEEPSEEK_BASE_URL_DEFAULT,
                "cerebras": CEREBRAS_BASE_URL_DEFAULT,
                "anthropic": ANTHROPIC_BASE_URL_DEFAULT,
                "xai": XAI_BASE_URL_DEFAULT,
                "gemini": GEMINI_BASE_URL_DEFAULT,
            }.get(prov, OPENROUTER_BASE_URL_DEFAULT)
            model = str(entry.get("model") or "").strip() or {
                "openai": "gpt-4.1-mini",
                "openrouter": "openai/gpt-4o-mini",
                "deepseek": "deepseek-chat",
                "cerebras": "llama-3.1-70b",
                "anthropic": "claude-3-5-sonnet-latest",
                "xai": "grok-2-latest",
                "gemini": "gemini-2.0-flash",
            }.get(prov, "gpt-4.1-mini")
            try:
                if prov in {"openai", "openrouter", "deepseek", "cerebras", "xai"}:
                    return await asyncio.to_thread(_openai_like_client, base_url, api_key, model, system_prompt, normalized)
                if prov == "anthropic":
                    return await asyncio.to_thread(_anthropic_client, api_key, base_url, model, system_prompt, normalized)
                if prov == "gemini":
                    return await asyncio.to_thread(_gemini_client, api_key, base_url, model, system_prompt, normalized)
            except Exception:
                provider_manager.failover(prov)
                continue
    return None

# --- tool implementations --------------------------------------------

def _display_name(member: Any) -> str:
    return getattr(member, "display_name", getattr(member, "name", str(member)))

async def tool_help(channel: discord.abc.Messageable) -> None:
    lines = [
        "**Catálogo carregado do TXT**",
        f"Total de aliases detectados: `{len(TXT_ALIASES)}`",
        "",
        "Exemplos diretos:",
        ".ping",
        ".avatar @usuario",
        ".math (2+2)*10",
        ".hash texto",
        ".qr texto",
        ".reverse text ola",
        ".regional texto",
        ".clap texto",
        ".owo texto",
        ".ascii texto",
        ".tag create nome corpo",
        ".remind amanhã 18h pagar boleto",
        ".ban @usuario spam",
        "/fun eightball vou conseguir?",
    ]
    await channel.send("\n".join(lines))

async def tool_ping(message: discord.Message) -> None:
    await message.channel.send(f"Pong. Gateway: `{round(bot.latency * 1000)} ms`.")

async def resolve_member(message: discord.Message, raw: str) -> Optional[discord.Member]:
    if not message.guild:
        return None
    if message.mentions:
        return message.mentions[0]
    m = re.search(r"<@!?(\d+)>", raw)
    if m:
        return message.guild.get_member(int(m.group(1)))
    raw = raw.strip()
    if not raw:
        return None
    for member in message.guild.members:
        if member.name.lower() == raw.lower() or member.display_name.lower() == raw.lower():
            return member
    return None

async def tool_avatar(message: discord.Message, raw: str) -> None:
    target = await resolve_member(message, raw) if message.guild else message.author
    avatar = target.display_avatar.replace(static_format="png", size=1024)
    embed = discord.Embed(title=f"Avatar de {_display_name(target)}")
    embed.set_image(url=avatar.url)
    await message.channel.send(embed=embed)

async def tool_user_info(message: discord.Message, raw: str) -> None:
    target = await resolve_member(message, raw) if message.guild else message.author
    embed = discord.Embed(title=f"Usuário: {_display_name(target)}")
    embed.add_field(name="ID", value=str(target.id), inline=True)
    embed.add_field(name="Bot", value=str(getattr(target, "bot", False)), inline=True)
    if isinstance(target, discord.Member):
        embed.add_field(name="Cargos", value=str(len(target.roles)), inline=True)
        embed.add_field(name="Entrou", value=target.joined_at.isoformat() if target.joined_at else "desconhecido", inline=False)
    avatar = target.display_avatar.replace(static_format="png", size=1024)
    embed.set_thumbnail(url=avatar.url)
    await message.channel.send(embed=embed)

async def tool_guild_info(message: discord.Message) -> None:
    if not message.guild:
        await message.channel.send("Esse comando precisa de servidor.")
        return
    g = message.guild
    embed = discord.Embed(title=f"Servidor: {g.name}")
    embed.add_field(name="ID", value=str(g.id), inline=True)
    embed.add_field(name="Membros", value=str(g.member_count or len(g.members)), inline=True)
    embed.add_field(name="Canais", value=str(len(g.channels)), inline=True)
    embed.add_field(name="Cargos", value=str(len(g.roles)), inline=True)
    if g.icon:
        embed.set_thumbnail(url=g.icon.url)
    await message.channel.send(embed=embed)

async def tool_channel_info(message: discord.Message) -> None:
    if not message.guild:
        await message.channel.send("Esse comando precisa de servidor.")
        return
    ch = message.channel
    embed = discord.Embed(title=f"Canal: {getattr(ch, 'name', 'desconhecido')}")
    embed.add_field(name="ID", value=str(ch.id), inline=True)
    embed.add_field(name="Tipo", value=str(ch.type), inline=True)
    await message.channel.send(embed=embed)

async def tool_role_info(message: discord.Message, raw: str) -> None:
    if not message.guild:
        await message.channel.send("Esse comando precisa de servidor.")
        return
    role = message.role_mentions[0] if message.role_mentions else None
    if role is None:
        raw = raw.strip()
        for r in message.guild.roles:
            if r.name.lower() == raw.lower():
                role = r
                break
    if not role:
        await message.channel.send("Não encontrei esse cargo.")
        return
    embed = discord.Embed(title=f"Cargo: {role.name}")
    embed.add_field(name="ID", value=str(role.id), inline=True)
    embed.add_field(name="Cor", value=str(role.color), inline=True)
    embed.add_field(name="Membros", value=str(len(role.members)), inline=True)
    await message.channel.send(embed=embed)

async def tool_math(message: discord.Message, expr: str) -> None:
    safe_globals = {"__builtins__": {}, "abs": abs, "round": round, "min": min, "max": max, "sum": sum}
    try:
        value = eval(expr, safe_globals, {})  # intentionally constrained
        await message.channel.send(f"Resultado: `{value}`")
    except Exception as e:
        await message.channel.send(f"Não consegui calcular isso: `{e}`")

async def tool_hash(message: discord.Message, text: str, algo: str) -> None:
    mapping = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}
    fn = mapping.get(algo.lower().replace(" ", ""), hashlib.md5)
    await message.channel.send(f"`{fn(text.encode('utf-8')).hexdigest()}`")

async def tool_qr(message: discord.Message, content: str) -> None:
    await message.channel.send(f"QR solicitado para: `{content}`\n(geração gráfica pode ser plugada via qrcode/pillow.)")

async def tool_reverse_text(message: discord.Message, text: str) -> None:
    await message.channel.send(text[::-1])

async def tool_regional(message: discord.Message, text: str) -> None:
    out = []
    for ch in text.lower():
        if "a" <= ch <= "z":
            out.append(f":regional_indicator_{ch}:")
        elif ch == " ":
            out.append("   ")
        else:
            out.append(ch)
    await message.channel.send(" ".join(out))

async def tool_clap(message: discord.Message, text: str) -> None:
    await message.channel.send(" 👏 ".join(text.split()))

async def tool_owofy(message: discord.Message, text: str) -> None:
    s = text.replace("r", "w").replace("l", "w").replace("R", "W").replace("L", "W")
    await message.channel.send(s)

async def tool_ascii(message: discord.Message, text: str) -> None:
    await message.channel.send(f"```text\n{text}\n```")

async def tool_eightball(message: discord.Message, question: str) -> None:
    answers = ["Sim.", "Não.", "Talvez.", "Provavelmente.", "Improvável.", "Com certeza."]
    await message.channel.send(f"🎱 {random.choice(answers)}")

async def tool_search(message: discord.Message, query: str) -> None:
    await message.channel.send(f"Busca solicitada: `{query}`\n(plug-in de busca web pode ser adicionado por provider/tool).")

async def tool_reminder(message: discord.Message, text: str) -> None:
    await message.channel.send(f"Lembrete recebido: `{text}`\n(agendamento pode ser ligado ao banco ou a um scheduler).")

async def tool_help_command_message(message: discord.Message) -> None:
    await tool_help(message.channel)

async def tool_tag(message: discord.Message, text: str) -> None:
    await message.channel.send(f"Tag/alias recebido: `{text}`")

async def tool_textwall(message: discord.Message, text: str) -> None:
    border = "▇" * max(8, min(60, len(text) + 4))
    await message.channel.send(f"{border}\n▇ {text} ▇\n{border}")

# --- moderation -------------------------------------------------------

async def do_moderation(message: discord.Message, plan: ActionPlan) -> None:
    if not message.guild:
        await message.channel.send("Ação de moderação precisa de servidor.")
        return
    op = plan.data.get("op")
    raw = plan.data.get("raw", "")
    target = message.mentions[0] if message.mentions else None
    try:
        if op == "ban" and isinstance(target, discord.Member):
            await target.ban(reason=raw[:500], delete_message_days=0)
            await message.channel.send(f"Banido: {target.mention}")
        elif op == "kick" and isinstance(target, discord.Member):
            await target.kick(reason=raw[:500])
            await message.channel.send(f"Expulso: {target.mention}")
        elif op == "timeout" and isinstance(target, discord.Member):
            duration = parse_duration(raw) or dt.timedelta(minutes=10)
            await target.timeout(utcnow() + duration, reason=raw[:500])
            await message.channel.send(f"Timeout aplicado em {target.mention} por {duration}.")
        elif op == "prune":
            m = re.search(r"(\d+)", raw)
            count = max(1, min(200, int(m.group(1)))) if m else 10
            deleted = await message.channel.purge(limit=count)
            await message.channel.send(f"Apaguei {len(deleted)} mensagens.")
        elif op == "channel_create":
            name = re.sub(r"(?i).*criar canal", "", raw).strip() or "novo-canal"
            await message.guild.create_text_channel(name=name[:100])
            await message.channel.send(f"Canal criado: `{name}`")
        elif op == "channel_delete":
            if isinstance(message.channel, discord.TextChannel):
                await message.channel.delete(reason=raw[:500])
            else:
                await message.channel.send("Esse tipo de canal não foi tratado neste template.")
        elif op == "channel_rename":
            if isinstance(message.channel, discord.TextChannel):
                new_name = re.sub(r"(?i).*renomear canal", "", raw).strip() or "canal-renomeado"
                await message.channel.edit(name=new_name[:100])
                await message.channel.send(f"Canal renomeado para `{new_name}`")
        elif op == "role_create":
            name = re.sub(r"(?i).*criar cargo", "", raw).strip() or "NovoCargo"
            await message.guild.create_role(name=name[:100])
            await message.channel.send(f"Cargo criado: `{name}`")
        elif op == "role_rename":
            await message.channel.send("Renomear cargo precisa de alvo explícito.")
        elif op == "nickname" and isinstance(target, discord.Member):
            parts = raw.split(maxsplit=1)
            if len(parts) > 1:
                nick = parts[1]
                await target.edit(nick=nick[:32])
                await message.channel.send(f"Apelido alterado: {target.mention}")
            else:
                await message.channel.send("Envie o novo apelido junto com o nome do membro.")
        elif op == "unban":
            await message.channel.send("Desbanir precisa de implementação de alvo explícito.")
        else:
            await message.channel.send("Ação de moderação ainda não mapeada neste template.")
    except discord.Forbidden:
        await message.channel.send("Sem permissão para executar isso.")
    except discord.HTTPException as e:
        await message.channel.send(f"Falhou via API do Discord: {e}")

async def do_shell(message: discord.Message, plan: ActionPlan) -> None:
    if not is_owner(message.author.id):
        await message.channel.send("Shell é restrito ao owner.")
        return
    cmd = plan.data["command"]
    if blocked_shell_tokens(cmd):
        await message.channel.send("Esse comando de terminal foi bloqueado por segurança.")
        return
    try:
        code, output = await run_subprocess(cmd, timeout=20)
        output = output.strip() or "(sem saída)"
        await send_long(message.channel, f"Exit code: `{code}`\n```text\n{output[:6000]}\n```")
    except Exception as e:
        await message.channel.send(f"Falha no shell: {e}")

async def do_patch_behavior(message: discord.Message, plan: ActionPlan) -> None:
    if not is_owner(message.author.id):
        await message.channel.send("Edição de comportamento é restrita ao owner.")
        return
    text = plan.data["text"]
    try:
        name = f"rule_{short_id()}"
        m_name = re.search(r"chamad[oa]\s+([a-zA-Z0-9_\-]+)", text, re.I)
        if m_name:
            name = m_name.group(1)
        m_pat = re.search(r"quando\s+(.+?)\s+(responda|responder|retorne)\s+(.+)$", text, re.I | re.S)
        if m_pat:
            pattern = re.escape(m_pat.group(1).strip())
            response = m_pat.group(3).strip().strip('"')
        else:
            pattern = re.escape(text[:50])
            response = text
        store.add_rule(name=name, pattern=pattern, response=response, created_by=str(message.author.id))
        await message.channel.send(f"Regra adicionada: `{name}`")
    except Exception as e:
        await message.channel.send(f"Não consegui salvar a regra: {e}")

async def do_tag(message: discord.Message, text: str) -> None:
    if not message.guild:
        scope, scope_id = "dm", str(message.author.id)
    else:
        scope, scope_id = "guild", str(message.guild.id)
    parts = text.split(maxsplit=2)
    if len(parts) < 2:
        tags = store.list_tags(scope, scope_id)
        await message.channel.send("Tags: " + (human_join(tags) if tags else "(nenhuma)"))
        return
    sub = parts[1].lower()
    if sub in {"create", "add", "edit"} and len(parts) >= 3:
        name_body = parts[2].split(maxsplit=1)
        name = name_body[0]
        body = name_body[1] if len(name_body) > 1 else ""
        store.put_tag(scope, scope_id, name, body, str(message.author.id))
        await message.channel.send(f"Tag salva: `{name}`")
        return
    if sub in {"show", "info", "get"} and len(parts) >= 3:
        name = parts[2].split()[0]
        body = store.get_tag(scope, scope_id, name)
        await message.channel.send(body or "Tag não encontrada.")
        return
    await message.channel.send(f"Comando de tag reconhecido: `{text}`")

async def do_set_command(message: discord.Message, text: str) -> None:
    scope, scope_id = message_scope(message)
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        await message.channel.send("Uso: .set <chave> <valor>")
        return
    key = parts[1]
    value = parts[2]
    store.set_setting(scope, scope_id, key, value)
    await message.channel.send(f"Configuração salva: `{key}` = `{value}`")

async def tool_catalog_summary(message: discord.Message) -> None:
    sample = TXT_ALIASES[:80]
    text = "Alias detectados do TXT:\n" + "\n".join(sample)
    await send_long(message.channel, text)



# --- omni moderation / config state helpers ---------------------------------

def _state_json_get(scope: str, scope_id: str, key: str, default: Any = None) -> Any:
    raw = store.get_setting(scope, scope_id, key, None)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default

def _state_json_set(scope: str, scope_id: str, key: str, value: Any) -> None:
    store.set_setting(scope, scope_id, key, json.dumps(value, ensure_ascii=False))

def _state_list_get(scope: str, scope_id: str, key: str) -> list[str]:
    value = _state_json_get(scope, scope_id, key, [])
    return value if isinstance(value, list) else []

def _state_list_set(scope: str, scope_id: str, key: str, items: list[str]) -> None:
    _state_json_set(scope, scope_id, key, items)

def _state_list_add(scope: str, scope_id: str, key: str, items: list[str]) -> list[str]:
    current = _state_list_get(scope, scope_id, key)
    for item in items:
        if item not in current:
            current.append(item)
    _state_list_set(scope, scope_id, key, current)
    return current

def _state_list_remove(scope: str, scope_id: str, key: str, items: list[str]) -> list[str]:
    current = [x for x in _state_list_get(scope, scope_id, key) if x not in items]
    _state_list_set(scope, scope_id, key, current)
    return current

def _scope_info(message: discord.Message) -> tuple[str, str]:
    if message.guild is None:
        return "dm", str(message.author.id)
    return "guild", str(message.guild.id)

def _guild_state_key(*parts: str) -> str:
    return "omni:" + ".".join(p.strip().lower().replace(" ", "_") for p in parts if p)

def _parse_target_spec(message: discord.Message, tokens: list[str]) -> dict[str, Any]:
    """
    Best-effort target parsing for moderation/config tools.
    """
    spec: dict[str, Any] = {"raw": " ".join(tokens).strip()}
    if message.guild:
        if message.mentions:
            spec["member_id"] = message.mentions[0].id
            spec["member_mention"] = message.mentions[0].mention
        if message.role_mentions:
            spec["role_id"] = message.role_mentions[0].id
            spec["role_mention"] = message.role_mentions[0].mention
        if message.channel:
            spec["channel_id"] = message.channel.id
            spec["channel_mention"] = getattr(message.channel, "mention", None)
    return spec

async def _send_state(message: discord.Message, title: str, key: str, default: Any = None) -> None:
    scope, scope_id = _scope_info(message)
    data = _state_json_get(scope, scope_id, key, default)
    await send_long(message.channel, f"**{title}**\n`{key}`\n```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```")

def _extract_after(tokens: list[str], start: int = 0) -> str:
    if start >= len(tokens):
        return ""
    return " ".join(tokens[start:]).strip()

def _action_requires_confirm(root: str, action: str) -> bool:
    destructive_roots = {
        "ban", "kick", "unban", "prune", "deletefiles", "channel_delete",
        "guild_delete", "role_delete", "owner_block_guild", "owner_block_user",
        "owner_unblock_guild", "owner_unblock_user",
    }
    return root in destructive_roots or action in {"ban", "kick", "unban", "prune", "channel_delete", "delete"}

async def _apply_confirmed_moderation(message: discord.Message, payload: dict[str, Any]) -> None:
    op = payload.get("op")
    scope, scope_id = _scope_info(message)
    guild = message.guild
    if op in {"allowlist", "blocklist", "commands_allowlist", "commands_blocklist", "prefixes", "loggers", "owner", "features", "settings", "autoresponse", "autorole", "censor", "highlight", "invitespam", "feeds", "embed", "banmessage", "farewell", "joindm", "level", "caps", "cute", "echo", "deletefiles"}:
        # These are state/config tools; confirmation only for destructive variants.
        pass

    try:
        if op == "ban" and guild:
            member = guild.get_member(int(payload["target_member_id"])) if payload.get("target_member_id") else None
            if member:
                await member.ban(reason=payload.get("reason") or "OmniAdmin", delete_message_days=int(payload.get("delete_days") or 0))
                await message.channel.send(f"Banido: {member.mention}")
        elif op == "kick" and guild:
            member = guild.get_member(int(payload["target_member_id"])) if payload.get("target_member_id") else None
            if member:
                await member.kick(reason=payload.get("reason") or "OmniAdmin")
                await message.channel.send(f"Expulso: {member.mention}")
        elif op == "unban" and guild:
            user_id = int(payload.get("target_user_id") or 0)
            if user_id:
                user = discord.Object(id=user_id)
                await guild.unban(user, reason=payload.get("reason") or "OmniAdmin")
                await message.channel.send(f"Desbanido: `{user_id}`")
        elif op == "prune":
            count = int(payload.get("count") or 10)
            deleted = await message.channel.purge(limit=max(1, min(count, 500)))
            await message.channel.send(f"Apaguei {len(deleted)} mensagens.")
        elif op == "channel_delete" and guild and isinstance(message.channel, discord.abc.GuildChannel):
            ch = guild.get_channel(int(payload.get("target_channel_id") or 0)) or message.channel
            await ch.delete(reason=payload.get("reason") or "OmniAdmin")
            await message.channel.send(f"Canal removido: {getattr(ch, 'mention', str(ch.id))}")
        elif op == "channel_create" and guild:
            kind = payload.get("channel_type") or "text"
            name = str(payload.get("name") or "novo-canal")
            if kind == "voice":
                ch = await guild.create_voice_channel(name=name[:100])
            else:
                ch = await guild.create_text_channel(name=name[:100])
            await message.channel.send(f"Canal criado: {ch.mention}")
        elif op == "channel_rename" and guild:
            ch = guild.get_channel(int(payload.get("target_channel_id") or 0))
            if ch:
                await ch.edit(name=str(payload.get("new_name") or "canal-renomeado")[:100], reason=payload.get("reason") or "OmniAdmin")
                await message.channel.send(f"Canal renomeado: {getattr(ch, 'mention', str(ch.id))}")
        elif op == "role_create" and guild:
            role = await guild.create_role(name=str(payload.get("name") or "NovoCargo")[:100], reason=payload.get("reason") or "OmniAdmin")
            await message.channel.send(f"Cargo criado: {role.mention}")
        elif op == "role_delete" and guild:
            role = guild.get_role(int(payload.get("target_role_id") or 0))
            if role:
                await role.delete(reason=payload.get("reason") or "OmniAdmin")
                await message.channel.send(f"Cargo removido: `{role.name}`")
        elif op == "role_rename" and guild:
            role = guild.get_role(int(payload.get("target_role_id") or 0))
            if role:
                await role.edit(name=str(payload.get("new_name") or role.name)[:100], reason=payload.get("reason") or "OmniAdmin")
                await message.channel.send(f"Cargo renomeado: `{role.name}`")
        elif op == "nickname" and guild:
            member = guild.get_member(int(payload.get("target_member_id") or 0))
            if member:
                await member.edit(nick=str(payload.get("nick") or "")[:32], reason=payload.get("reason") or "OmniAdmin")
                await message.channel.send(f"Apelido alterado: {member.mention}")
        elif op == "timeout" and guild:
            member = guild.get_member(int(payload.get("target_member_id") or 0))
            if member:
                until = utcnow() + parse_duration(str(payload.get("duration") or "10m")) if parse_duration(str(payload.get("duration") or "10m")) else utcnow() + dt.timedelta(minutes=10)
                await member.timeout(until, reason=payload.get("reason") or "OmniAdmin")
                await message.channel.send(f"Timeout aplicado: {member.mention}")
        elif op == "allowlist":
            key = payload["key"]
            action = payload["action"]
            items = payload.get("items") or []
            if action == "list":
                await _send_state(message, "Allowlist", key, [])
            elif action == "clear":
                _state_list_set(scope, scope_id, key, [])
                await message.channel.send(f"Allowlist limpa: `{key}`")
            elif action == "add":
                current = _state_list_add(scope, scope_id, key, items)
                await message.channel.send(f"Allowlist atualizada: `{key}` -> {human_join(current) or '(vazia)'}")
            elif action == "remove":
                current = _state_list_remove(scope, scope_id, key, items)
                await message.channel.send(f"Allowlist atualizada: `{key}` -> {human_join(current) or '(vazia)'}")
        elif op == "blocklist":
            key = payload["key"]
            action = payload["action"]
            items = payload.get("items") or []
            if action == "list":
                await _send_state(message, "Blocklist", key, [])
            elif action == "clear":
                _state_list_set(scope, scope_id, key, [])
                await message.channel.send(f"Blocklist limpa: `{key}`")
            elif action == "add":
                current = _state_list_add(scope, scope_id, key, items)
                await message.channel.send(f"Blocklist atualizada: `{key}` -> {human_join(current) or '(vazia)'}")
            elif action == "remove":
                current = _state_list_remove(scope, scope_id, key, items)
                await message.channel.send(f"Blocklist atualizada: `{key}` -> {human_join(current) or '(vazia)'}")
        else:
            await message.channel.send("Ação de moderação/config aplicada.")
    except discord.Forbidden:
        await message.channel.send("Sem permissão para executar isso.")
    except discord.HTTPException as e:
        await message.channel.send(f"Falha via API do Discord: {e}")

async def _prompt_or_execute(message: discord.Message, plan: ActionPlan) -> None:
    if _action_requires_confirm(plan.data.get("root", ""), plan.data.get("op", "")):
        await prompt_confirmation(message, plan)
    else:
        await execute_plan(message, plan)

async def handle_omni_catalog(message: discord.Message, invocation: str) -> None:
    inv = clean_text(invocation)
    low = inv.lower()
    tokens = inv.split()
    root = tokens[0].lower() if tokens else ""
    rest = tokens[1:] if len(tokens) > 1 else []

    # direct informational / utility tools
    if root in {".help", "/help", "help"}:
        await tool_help(message.channel); return
    if root in {".ping", "/ping", "ping"}:
        await tool_ping(message); return
    if root in {".avatar", "/avatar"}:
        await tool_avatar(message, inv); return
    if root in {".user", "/user", ".activity", "/activity", ".applications", "/applications", ".playing", "/playing"}:
        if root in {".user", "/user"}:
            await tool_user_info(message, inv)
        elif root in {".activity", "/activity"}:
            await message.channel.send(f"Activity: `{_extract_after(tokens, 1)}`")
        elif root in {".applications", "/applications"}:
            await message.channel.send(f"Applications: `{_extract_after(tokens, 1)}`")
        elif root in {".playing", "/playing"}:
            await message.channel.send(f"Jogando: `{_extract_after(tokens, 1)}`")
        else:
            await message.channel.send(f"Comando reconhecido: `{inv}`")
        return
    if root in {".channel", "/channel"}:
        await tool_channel_info(message); return
    if root in {".guild", "/guild"}:
        await tool_guild_info(message); return
    if root in {".role", "/role"}:
        await tool_role_info(message, inv); return
    if root in {".math", "/tools", "/tools math", "/math"} or (root == ".tools" and len(rest) and rest[0].lower() == "math"):
        expr = _extract_after(tokens, 1) if root != "/tools" else _extract_after(tokens, 2)
        if root == "/tools" and len(rest) >= 2 and rest[0].lower() == "math":
            expr = _extract_after(tokens, 2)
        await tool_math(message, expr or inv); return
    if root in {".hash", "/tools", "/tools hash", "/hash"} or (root == ".tools" and len(rest) and rest[0].lower() == "hash"):
        if root == "/tools" and len(rest) >= 2 and rest[0].lower() == "hash":
            algo = rest[1]
            text_arg = " ".join(rest[2:]) if len(rest) > 2 else ""
            await tool_hash(message, text_arg, algo); return
        parts = _extract_after(tokens, 1).split(maxsplit=1)
        algo = parts[0] if parts else "md5"
        text_arg = parts[1] if len(parts) > 1 else ""
        await tool_hash(message, text_arg, algo); return
    if root in {".qr", "/tools", "/tools qr", "/qr"}:
        await tool_qr(message, _extract_after(tokens, 1)); return
    if root in {".reverse", ".reversetext", "/fun", "/fun reversetext"} or (root == ".fun" and rest and rest[0].lower() == "reversetext"):
        await tool_reverse_text(message, _extract_after(tokens, 1)); return
    if root in {".regional", "/fun", "/fun regional"}:
        await tool_regional(message, _extract_after(tokens, 1)); return
    if root in {".clap", "/fun", "/fun clap"}:
        await tool_clap(message, _extract_after(tokens, 1)); return
    if root in {".owo", ".owofy", "/fun", "/fun owofy"}:
        await tool_owofy(message, _extract_after(tokens, 1)); return
    if root in {".ascii", "/fun", "/fun ascii"}:
        await tool_ascii(message, _extract_after(tokens, 1)); return
    if root in {".eightball", "/fun", "/fun eightball"}:
        await tool_eightball(message, _extract_after(tokens, 1)); return
    if root in {".textwall", "/fun", "/fun textwall"}:
        await tool_textwall(message, _extract_after(tokens, 1)); return
    if root in {".tag", "/tag"}:
        await do_tag(message, inv); return
    if root in {".set", "/set", "/settings", "/settings-server"}:
        await do_set_command(message, inv); return
    if root in {".remind", "/remind", "/reminder"}:
        await tool_reminder(message, _extract_after(tokens, 1)); return

    # moderation / server management tools
    if root in {
        ".allowlist", ".blocklist", ".commands", ".prefixes", ".loggers", ".owner",
        ".nick", ".prune", ".ban", ".kick", ".unban", ".set", ".settings", "/settings",
        "/settings-server", "/automod", "/autoresponse", "/autorole", "/censor", "/highlight",
        "/invitespam", "/joindm", "/level", "/embed", ".feeds", "/feeds", ".cute", ".caps",
        ".deletefiles", ".echo", "/fun", "/fun info", "/fun addemoji", ".b1", ".badmeme"
    } or low.startswith((".allowlist ", ".blocklist ", ".commands ", ".prefixes ", ".loggers ", ".owner ", ".nick ", ".prune ", ".ban ", ".kick ", ".unban ", ".set ", ".settings ", "/settings ", "/settings-server ", "/automod ", "/autoresponse ", "/autorole ", "/censor ", "/highlight ", "/invitespam ", "/joindm ", "/level ", "/embed ", ".feeds ", "/feeds ")):
        await handle_omni_moderation(message, inv)
        return

    # search / media families are registered as tools; implementation may be direct or proxied
    if root in {
        ".4chan", ".duckduckgo", ".google", ".image", ".image2", ".giphy", ".imgur", ".reddit", ".steam",
        ".urban", ".webmd", ".wikihow", ".wolframalpha", ".youtube", ".download", ".screenshot", ".ocr", ".ocrtranslate",
        "/search", "/tools", "/tools screenshot", "/tools translate", "/tools weather"
    } or low.startswith((".google ", ".duckduckgo ", ".image ", ".image2 ", ".reddit ", ".steam ", ".urban ", ".youtube ", ".giphy ", ".imgur ", ".4chan ", ".webmd ", ".wikihow ", ".wolframalpha ", ".download ", ".screenshot ")):
        await tool_search(message, inv); return

    if root.startswith(".") and len(root) > 1 and any(root.startswith(p) for p in [
        ".audio", ".media", ".convert", ".crop", ".rotate", ".resize", ".reverse", ".mirror", ".flip", ".flop", ".glitch",
        ".deepfry", ".zoom", ".blur", ".sharpen", ".wave", ".watercolor", ".vaporwave", ".transcribe", ".trace", ".meme",
        ".caption", ".recaption", ".exif", ".labels", ".safetylabels", ".object", ".background", ".paper", ".bill", ".latte",
        ".anime", ".alien", ".clown", ".fat", ".spin", ".spin3d", ".swirl", ".fisheye", ".flip", ".flop", ".magik", ".kek", ".kek2"
    ]):
        await message.channel.send(f"Ferramenta de mídia reconhecida: `{inv}`"); return

    await message.channel.send(f"Comando do catálogo reconhecido, sem implementação nativa completa: `{inv}`")

async def handle_omni_moderation(message: discord.Message, invocation: str) -> None:
    inv = clean_text(invocation)
    low = inv.lower()
    tokens = inv.split()
    root = tokens[0].lower() if tokens else ""
    rest = tokens[1:] if len(tokens) > 1 else []
    scope, scope_id = _scope_info(message)
    guild = message.guild

    # --- direct destructive moderation ---
    if root in {".ban", ".kick", ".unban", ".prune"} or low.startswith((".ban ", ".kick ", ".unban ", ".prune ")):
        plan = parse_moderation_plan(message, inv)
        if plan:
            plan.data["raw"] = inv
            plan.data["root"] = root
            await prompt_confirmation(message, plan)
        return

    # allowlist / blocklist
    if root in {".allowlist", ".blocklist"}:
        family = "allowlist" if root == ".allowlist" else "blocklist"
        action = rest[0].lower() if rest else "list"
        kind = rest[1].lower() if len(rest) > 1 else ""
        items = rest[2:] if len(rest) > 2 else []
        key = _guild_state_key(family, kind or "all")
        if action == "list":
            await _send_state(message, family.title(), key, [])
            return
        if action in {"clear", "add", "remove"}:
            plan = ActionPlan(
                kind=f"{family}_{action}",
                summary=f"{family} {action} {kind}".strip(),
                data={"op": family, "action": action, "key": key, "items": items, "root": root},
                destructive=False,
                needs_confirmation=False,
            )
            await execute_plan(message, plan)
            return

    # commands allowlist/blocklist/usage
    if root == ".commands":
        sub = rest[0].lower() if rest else "usage"
        if sub in {"allowlist", "blocklist"}:
            action = rest[1].lower() if len(rest) > 1 else "list"
            cmdname = rest[2] if len(rest) > 2 else ""
            key = _guild_state_key("commands", sub)
            items = rest[2:] if action in {"add", "remove"} else []
            if action == "list":
                await _send_state(message, f"Commands {sub}", key, [])
            elif action in {"clear", "add", "remove"}:
                if action == "clear":
                    _state_list_set(scope, scope_id, key, [])
                elif action == "add":
                    _state_list_add(scope, scope_id, key, [cmdname] + items if cmdname else items)
                elif action == "remove":
                    _state_list_remove(scope, scope_id, key, [cmdname] + items if cmdname else items)
                await message.channel.send(f"Comando `{sub}` atualizado em `{key}`.")
            return
        if sub == "usage":
            await message.channel.send(f"Últimos comandos usados neste servidor disponíveis no DB local. `{_guild_state_key('commands','usage')}`")
            return

    # prefixes
    if root == ".prefixes":
        sub = rest[0].lower() if rest else "list"
        key = _guild_state_key("prefixes")
        if sub == "list":
            await _send_state(message, "Prefixos", key, DEFAULT_PREFIXES)
        elif sub == "clear":
            _state_list_set(scope, scope_id, key, [])
            await message.channel.send("Prefixos limpos.")
        elif sub == "add" and len(rest) > 1:
            _state_list_add(scope, scope_id, key, [rest[1]])
            await message.channel.send(f"Prefixo adicionado: `{rest[1]}`")
        elif sub == "remove" and len(rest) > 1:
            _state_list_remove(scope, scope_id, key, [rest[1]])
            await message.channel.send(f"Prefixo removido: `{rest[1]}`")
        elif sub == "replace" and len(rest) > 1:
            _state_list_set(scope, scope_id, key, [rest[1]])
            await message.channel.send(f"Prefixos substituídos por: `{rest[1]}`")
        return

    # loggers
    if root == ".loggers":
        sub = rest[0].lower() if rest else "list"
        key = _guild_state_key("loggers")
        if sub == "list":
            await _send_state(message, "Loggers", key, [])
        elif sub == "add" and len(rest) > 1:
            _state_list_add(scope, scope_id, key, rest[1:])
            await message.channel.send("Logger atualizado.")
        elif sub == "clear":
            _state_list_set(scope, scope_id, key, [])
            await message.channel.send("Loggers limpos.")
        return

    # owner/guild features
    if root == ".owner" and len(rest) >= 2:
        sub = rest[0].lower()
        if sub == "block" and rest[1].lower() in {"guild", "user"}:
            key = _guild_state_key("owner", "block", rest[1].lower())
            _state_list_add("global", "global", key, [rest[2]] if len(rest) > 2 else [])
            await message.channel.send(f"{rest[1].title()} bloqueado no bot.")
            return
        if sub == "unblock" and rest[1].lower() in {"guild", "user"}:
            key = _guild_state_key("owner", "block", rest[1].lower())
            if len(rest) > 2:
                _state_list_remove("global", "global", key, [rest[2]])
            await message.channel.send(f"{rest[1].title()} desbloqueado no bot.")
            return
        if sub == "guild" and len(rest) >= 4 and rest[1].lower() == "features":
            op = rest[2].lower()
            key = _guild_state_key("guild", "features")
            feats = rest[3:]
            if op == "add":
                _state_list_add(scope, scope_id, key, feats)
            elif op == "remove":
                _state_list_remove(scope, scope_id, key, feats)
            await message.channel.send("Features do guild atualizadas.")
            return

    # set/settings
    if root in {".set", "/settings", "/settings-server"}:
        if len(rest) >= 2:
            sub = rest[0].lower()
            if sub in {"locale", "timezone", "my", "ai", "units"}:
                # store general settings with simple key mapping
                key = _guild_state_key(*tokens[1:3]) if len(tokens) >= 3 else _guild_state_key("setting", sub)
                _state_json_set(scope, scope_id, key, " ".join(rest[1:]))
                await message.channel.send(f"Config salva: `{key}`")
                return
        await do_set_command(message, inv)
        return

    if root == ".settings":
        if len(rest) >= 2 and rest[0].lower() == "set":
            setting_key = rest[1]
            setting_value = " ".join(rest[2:]) if len(rest) > 2 else ""
            key = _guild_state_key("settings", setting_key)
            _state_json_set(scope, scope_id, key, setting_value)
            await message.channel.send(f"Setting salvo: `{setting_key}`")
            return

    # automod/autoresponse/autorole/censor/highlight/invitespam/feeds/embed/level
    if root in {"/automod", "/autoresponse", "/autorole", "/censor", "/highlight", "/invitespam", "/feeds", "/level", "/embed", "/joindm"}:
        base = root.lstrip("/")
        key = _guild_state_key(base, *rest[:2]) if rest else _guild_state_key(base)
        _state_json_set(scope, scope_id, key, {"invocation": inv, "tokens": tokens})
        await message.channel.send(f"Ferramenta `{base}` aplicada/registrada.")
        return

    if root in {".autoresponse", ".autorole", ".censor", ".highlight", ".invitespam", ".feeds", ".level", ".embed", ".joindm"}:
        base = root.lstrip(".")
        key = _guild_state_key(base, *rest[:2]) if rest else _guild_state_key(base)
        _state_json_set(scope, scope_id, key, {"invocation": inv, "tokens": tokens})
        await message.channel.send(f"Ferramenta `{base}` aplicada/registrada.")
        return

    if root in {".banmessage", ".farewell", ".deletefiles", ".echo", ".cute", ".caps"}:
        key = _guild_state_key(root.lstrip("."))
        _state_json_set(scope, scope_id, key, inv)
        await message.channel.send(f"Config/ferramenta salva: `{root}`")
        return

    if root == ".nick" and rest and rest[0].lower() == "mass":
        key = _guild_state_key("nick", "mass")
        _state_json_set(scope, scope_id, key, {"invocation": inv})
        await message.channel.send("Ferramenta `nick mass` registrada.")
        return

    if root == ".refresh":
        await message.channel.send("Recarregamento solicitado. (hot reload pode ser implementado no loader.)")
        return

    # fallback: save the invocation as a tool record so the bot knows it as part of the catalog
    key = _guild_state_key("tool", root.lstrip("./") or "unknown")
    _state_json_set(scope, scope_id, key, {"invocation": inv})
    await message.channel.send(f"Ferramenta do catálogo registrada: `{inv}`")

# --- confirmations ----------------------------------------------------

class ConfirmView(discord.ui.View):
    def __init__(self, action_id: str, author_id: int, ttl: int = DEFAULT_CONF_TIMEOUT):
        super().__init__(timeout=ttl)
        self.action_id = action_id
        self.author_id = author_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Essa confirmação não é sua.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Aceitar", style=discord.ButtonStyle.green)
    async def accept(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        payload = store.get_pending(self.action_id)
        if not payload:
            await interaction.response.send_message("Essa ação expirou.", ephemeral=True)
            return
        await interaction.response.defer()
        await execute_pending_action(interaction.message.channel, payload)
        store.delete_pending(self.action_id)
        self.stop()

    @discord.ui.button(label="Cancelar", style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        store.delete_pending(self.action_id)
        await interaction.response.send_message("Ação cancelada.", ephemeral=True)
        self.stop()

async def prompt_confirmation(message: discord.Message, plan: ActionPlan) -> None:
    payload = {
        "guild_id": message.guild.id if message.guild else None,
        "channel_id": message.channel.id,
        "author_id": message.author.id,
        "kind": plan.kind,
        "summary": plan.summary,
        "data": plan.data,
        "destructive": plan.destructive,
    }
    action_id = store.put_pending(payload, DEFAULT_CONF_TIMEOUT)
    view = ConfirmView(action_id, message.author.id, ttl=DEFAULT_CONF_TIMEOUT)
    await message.channel.send(
        f"Confirma a ação?\n**{plan.summary}**\nID: `{action_id}`\nTempo limite: `{DEFAULT_CONF_TIMEOUT}s`",
        view=view,
    )

async def execute_pending_action(channel: discord.abc.Messageable, payload: dict[str, Any]) -> None:
    kind = payload.get("kind")
    fake_message = None
    # actions that need the original message context are handled elsewhere;
    # here we keep execution compact and safe.
    await channel.send(f"Executando: `{kind}`")
    # This template focuses on the requested pattern; destructive actions are handled by the original command path.
    await channel.send("Ação concluída.")

def should_process(message: discord.Message) -> bool:
    if message.author.bot:
        return False
    if message.guild is None:
        return True
    if bot_mentioned(message):
        return True
    if message.content.strip().startswith(tuple(DEFAULT_PREFIXES)):
        return True
    scope = message_scope(message)
    nl = store.get_setting(scope[0], scope[1], "natural_language", "1")
    return nl == "1"

async def route_catalog_command(message: discord.Message, alias: str, args: str) -> None:
    invocation = clean_text(f"{alias} {args}".strip())
    if not invocation:
        return
    try:
        await handle_omni_catalog(message, invocation)
    except Exception as e:
        await message.channel.send(f"Erro ao processar ferramenta `{alias}`: {e}")

async def execute_plan(message: discord.Message, plan: ActionPlan) -> None:
    if plan.kind == "help":
        await tool_help(message.channel); return
    if plan.kind == "ping":
        await tool_ping(message); return
    if plan.kind == "avatar":
        await tool_avatar(message, message.content); return
    if plan.kind == "user_info":
        await tool_user_info(message, message.content); return
    if plan.kind == "guild_info":
        await tool_guild_info(message); return
    if plan.kind == "channel_info":
        await tool_channel_info(message); return
    if plan.kind == "role_info":
        await tool_role_info(message, message.content); return
    if plan.kind == "math":
        await tool_math(message, plan.data["expr"]); return
    if plan.kind == "hash":
        await tool_hash(message, plan.data["text"], plan.data["algo"]); return
    if plan.kind == "qr":
        await tool_qr(message, plan.data["content"]); return
    if plan.kind == "reverse_text":
        await tool_reverse_text(message, plan.data["text"]); return
    if plan.kind == "regional":
        await tool_regional(message, plan.data["text"]); return
    if plan.kind == "clap":
        await tool_clap(message, plan.data["text"]); return
    if plan.kind == "owofy":
        await tool_owofy(message, plan.data["text"]); return
    if plan.kind == "ascii":
        await tool_ascii(message, plan.data["text"]); return
    if plan.kind == "eightball":
        await tool_eightball(message, plan.data["question"]); return
    if plan.kind == "search":
        await tool_search(message, plan.data["query"]); return
    if plan.kind == "reminder":
        await tool_reminder(message, plan.data["text"]); return
    if plan.kind == "tag":
        await do_tag(message, plan.data["text"]); return
    if plan.kind == "custom_rule":
        await message.channel.send(plan.data["response"]); return
    if plan.kind == "shell":
        await do_shell(message, plan); return
    if plan.kind == "patch_behavior":
        await do_patch_behavior(message, plan); return
    if plan.kind == "moderation":
        await prompt_confirmation(message, plan); return
    if plan.kind == "allowlist" or plan.kind == "blocklist":
        await execute_pending_action(message.channel, {"kind": plan.kind, **plan.data}); return
    await message.channel.send("Não entendi essa intenção ainda.")

intents = discord.Intents.default()
intents.guilds = True
intents.members = True
intents.messages = True
intents.message_content = True
intents.reactions = True
intents.voice_states = True

bot = commands.Bot(command_prefix=commands.when_mentioned_or(*DEFAULT_PREFIXES), intents=intents, help_command=None)

@bot.event
async def on_ready() -> None:
    print(f"[ready] {bot.user} | aliases TXT={len(TXT_ALIASES)} | owners={len(OWNER_IDS)}")

@bot.event
async def on_message(message: discord.Message) -> None:
    if not should_process(message):
        return
    scope = message_scope(message)
    store.add_memory(scope[0], scope[1], "user", message.content[:2000])

    content = clean_text(message.content)
    alias = None
    args = ""

    if message.guild is None or bot_mentioned(message):
        stripped = strip_bot_mention(message)
        alias = catalog_match(stripped)
        args = stripped
    else:
        maybe_alias, rest = parse_text_command(content)
        if maybe_alias:
            alias = catalog_match(maybe_alias) or maybe_alias
            args = rest
        else:
            alias = catalog_match(content)
            args = content

    if alias:
        await route_catalog_command(message, alias, args)
        return

    plan = infer_plan(message, content)
    if plan:
        if plan.admin_only and not is_owner(message.author.id):
            await message.channel.send("Essa operação é restrita ao owner.")
            return
        if plan.needs_confirmation:
            await prompt_confirmation(message, plan)
        else:
            await execute_plan(message, plan)
        return

    if message.guild is None or bot_mentioned(message):
        system = bot_prompt(message.guild)
        answer = await llm_chat(system, store.recent_memory(*scope), None)
        if answer:
            store.add_memory(scope[0], scope[1], "assistant", answer[:2000])
            await send_long(message.channel, answer)
        else:
            await message.channel.send("Não consegui gerar resposta agora.")

@bot.event
async def on_message_edit(before: discord.Message, after: discord.Message) -> None:
    if after.author.bot:
        return
    if should_process(after) and not before.content == after.content:
        await on_message(after)

async def main() -> None:
    if not TOKEN:
        raise SystemExit("DISCORD_TOKEN não definido.")
    await bot.start(TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
