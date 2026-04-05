#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omni Admin Bot — single-file, tool-first Discord bot.

Principles:
- Mention / DM / natural-language driven.
- Legacy dot/slash commands are only compatibility shims; the public interface is tool-based.
- Loads the TXT catalog next to this file and turns it into discoverable aliases.
- Stores state in SQLite.
- Supports a tool registry so new capabilities can be added without splitting the file.

Environment variables of note:
- DISCORD_TOKEN: Discord bot token.
- OWNER_IDS: comma-separated user IDs with full access.
- TXT_CATALOG_PATH: optional path to the command catalog TXT.
- ALLOW_LEGACY_PREFIXES: set to 1 to accept . / ! compatibility prefixes.
- ALLOW_SHELL: set to 1 to enable the shell tool.
- LLM_PROVIDER_ORDER: comma-separated provider priority.
- OPENAI_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / XAI_API_KEY / GEMINI_API_KEY / etc.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import dataclasses
import datetime as dt
import hashlib
import importlib
import inspect
import io
import json
import os
import random
import re
import shlex
import sqlite3
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

REQUIRED_PACKAGES = [
    "discord.py>=2.4.0",
    "requests>=2.31.0",
    "python-dateutil>=2.9.0.post0",
]

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

def _ensure_packages() -> None:
    module_map = {"discord.py": "discord", "requests": "requests", "python-dateutil": "dateutil"}
    missing: list[str] = []
    for pkg in REQUIRED_PACKAGES:
        name = pkg.split(">=")[0]
        module_name = module_map.get(name, name.replace("-", "_"))
        try:
            importlib.import_module(module_name)
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
TXT_CATALOG_PATH = Path(os.getenv("TXT_CATALOG_PATH", str(BASE_DIR / "Interface do sistema shared text (2).txt")))

TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
OWNER_IDS = {int(x) for x in re.split(r"[,\s]+", os.getenv("OWNER_IDS", "").strip()) if x.isdigit()}
ALLOW_SHELL = _env_bool("ALLOW_SHELL", "0")
ALLOW_LEGACY_PREFIXES = _env_bool("ALLOW_LEGACY_PREFIXES", "0")
CHAT_ENABLED = _env_bool("CHAT_ENABLED", "1")
DEFAULT_CONF_TIMEOUT = _env_int("CONFIRM_TIMEOUT_SECONDS", "300")

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
    return re.sub(r"\s+", " ", normalize_ws(text)).strip()

def human_join(items: list[str]) -> str:
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " e " + items[-1]

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
        "rm -rf", "rm -fr", "mkfs", "shutdown", "reboot", "poweroff", "curl ", "wget ",
        "| sh", "|bash", "sudo", "dd ", "chmod 777", ":(){", "forkbomb", "kill -9",
        "taskkill", "format c:",
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
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
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
            "ON CONFLICT(name) DO UPDATE SET pattern=excluded.pattern, response=excluded.response, "
            "created_by=excluded.created_by, created_at=excluded.created_at",
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
            "ON CONFLICT(scope, scope_id, name) DO UPDATE SET content=excluded.content, created_by=excluded.created_by, "
            "created_at=excluded.created_at",
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

    def add_note(self, scope: str, scope_id: str, author_id: str, content: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO notes(scope, scope_id, author_id, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (scope, scope_id, author_id, content, utcnow_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_notes(self, scope: str, scope_id: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, author_id, content, created_at FROM notes WHERE scope=? AND scope_id=? ORDER BY id DESC LIMIT ?",
            (scope, scope_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, note_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        self.conn.commit()
        return cur.rowcount > 0


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
            "ollama": [{"name": "ollama-local", "base_url": LOCAL_OLLAMA_BASE_URL, "model": LOCAL_OLLAMA_MODEL}],
            "openrouter": [{"name": "openrouter-1", "api_key": "", "base_url": OPENROUTER_BASE_URL_DEFAULT, "model": "openai/gpt-4o-mini"}],
            "openai": [{"name": "openai-1", "api_key": "", "base_url": OPENAI_BASE_URL_DEFAULT, "model": "gpt-4.1-mini"}],
            "deepseek": [{"name": "deepseek-1", "api_key": "", "base_url": DEEPSEEK_BASE_URL_DEFAULT, "model": "deepseek-chat"}],
            "cerebras": [{"name": "cerebras-1", "api_key": "", "base_url": CEREBRAS_BASE_URL_DEFAULT, "model": "llama-3.1-70b"}],
            "anthropic": [{"name": "anthropic-1", "api_key": "", "base_url": ANTHROPIC_BASE_URL_DEFAULT, "model": "claude-3-5-sonnet-latest"}],
            "xai": [{"name": "xai-1", "api_key": "", "base_url": XAI_BASE_URL_DEFAULT, "model": "grok-2-latest"}],
            "gemini": [{"name": "gemini-1", "api_key": "", "base_url": GEMINI_BASE_URL_DEFAULT, "model": "gemini-2.0-flash"}],
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
        if entries:
            self.indices[provider] = (self.indices.get(provider, 0) + 1) % len(entries)


provider_manager = ProviderManager(KEYS_CONFIG)


def catalog_from_txt(path: Path) -> list[str]:
    if not path.exists():
        return []
    aliases: list[str] = []
    seen: set[str] = set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = normalize_ws(raw)
        if not line:
            continue
        if line in {
            "Commands", "Type", "Prefixed", "Slash", "Context Menu Message", "Context Menu User",
            "Category", "All", "Fun", "Image Manipulation", "Informational", "Moderation",
            "Bot Owner Only", "Say", "Search", "Server Settings", "Tools", "Utilities",
        }:
            continue
        if not (line.startswith(".") or line.startswith("/")):
            continue
        for match in re.finditer(r'(?<!\w)([./][A-Za-z0-9][\w-]*(?:\s+[A-Za-z0-9][\w-]*)*)', line):
            alias = match.group(1).strip().lower()
            alias = re.sub(r"\s+", " ", alias)
            alias = alias.rstrip(".,;:?")
            if alias not in seen:
                seen.add(alias)
                aliases.append(alias)
    return aliases


TXT_ALIASES = catalog_from_txt(TXT_CATALOG_PATH)
TXT_ALIASES_SET = set(TXT_ALIASES)

def canonical_from_legacy(alias: str) -> str:
    alias = clean_text(alias).lower()
    if alias.startswith((".", "/", "!")):
        alias = alias[1:].lstrip()
    alias = re.sub(r"\s+", " ", alias)
    return alias

CANONICAL_TO_ALIASES: dict[str, set[str]] = {}
for alias in TXT_ALIASES:
    canonical = canonical_from_legacy(alias)
    CANONICAL_TO_ALIASES.setdefault(canonical, set()).add(alias)

def guess_canonical(text: str) -> Optional[str]:
    t = clean_text(text).lower()
    t = t.lstrip()
    if t in TXT_ALIASES_SET:
        return canonical_from_legacy(t)
    for alias in sorted(TXT_ALIASES, key=len, reverse=True):
        if t == alias or t.startswith(alias + " ") or alias.startswith(t + " "):
            return canonical_from_legacy(alias)
    return None


def split_known_tool(text: str) -> tuple[Optional[str], str]:
    """
    Return the longest registered tool name that prefixes the text.
    This allows multiword tools such as "commands allowlist" or "audio put mix".
    """
    t = clean_text(text).lower()
    if not t:
        return None, ""
    names = sorted(registry.tools.keys(), key=len, reverse=True)
    for name in names:
        if t == name:
            return name, ""
        if t.startswith(name + " "):
            return name, t[len(name):].strip()
    return None, ""

def parse_text_command(content: str) -> tuple[Optional[str], str]:
    txt = clean_text(content)
    if not txt:
        return None, ""
    lowered = txt.lower()
    for lead in ("tool ", "tools ", "use ", "invoke ", "run "):
        if lowered.startswith(lead):
            rest = txt[len(lead):].strip()
            if not rest:
                return None, ""
            name, args = split_known_tool(rest)
            if name:
                return name, args
            parts = rest.split(maxsplit=1)
            return parts[0].lower(), parts[1] if len(parts) > 1 else ""
    if ALLOW_LEGACY_PREFIXES and txt[0] in {".", "/", "!"}:
        rest = txt[1:].strip()
        if not rest:
            return None, ""
        name, args = split_known_tool(rest)
        if name:
            return name, args
        parts = rest.split(maxsplit=1)
        return parts[0].lower(), parts[1] if len(parts) > 1 else ""
    canonical = guess_canonical(txt)
    if canonical:
        name, args = split_known_tool(canonical)
        return (name or canonical), args
    name, args = split_known_tool(txt)
    if name:
        return name, args
    return None, txt

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

def safe_short(text: str, limit: int = 1800) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 20] + " …[cortado]"

def chunk_text(text: str, size: int = 1900) -> list[str]:
    text = str(text)
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


@dataclass
class ToolContext:
    bot: "OmniBot"
    message: discord.Message
    name: str
    args: str
    scope: str
    scope_id: str

    @property
    def guild(self) -> Optional[discord.Guild]:
        return self.message.guild

    @property
    def channel(self) -> discord.abc.Messageable:
        return self.message.channel

    @property
    def author(self) -> discord.User | discord.Member:
        return self.message.author


ToolHandler = Callable[[ToolContext], Awaitable[str] | str]


@dataclass
class ToolSpec:
    name: str
    description: str
    category: str = "utility"
    aliases: set[str] = field(default_factory=set)
    destructive: bool = False
    admin_only: bool = False
    handler: Optional[ToolHandler] = None


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}
        self.alias_map: dict[str, str] = {}
        self.categories: dict[str, set[str]] = {}

    def register(self, spec: ToolSpec) -> None:
        key = spec.name.lower()
        self.tools[key] = spec
        self.alias_map[key] = key
        self.categories.setdefault(spec.category, set()).add(key)
        for alias in spec.aliases:
            self.alias_map[alias.lower()] = key

    def resolve(self, name: str) -> Optional[ToolSpec]:
        key = clean_text(name).lower()
        key = self.alias_map.get(key, key)
        return self.tools.get(key)

    def list_tools(self) -> list[ToolSpec]:
        return [self.tools[k] for k in sorted(self.tools)]

registry = ToolRegistry()


def tool(name: str, *, description: str, category: str = "utility", aliases: Optional[set[str]] = None, destructive: bool = False, admin_only: bool = False):
    def decorator(func: ToolHandler):
        registry.register(
            ToolSpec(
                name=name,
                description=description,
                category=category,
                aliases=aliases or set(),
                destructive=destructive,
                admin_only=admin_only,
                handler=func,
            )
        )
        return func
    return decorator


def is_owner(user_id: int) -> bool:
    return user_id in OWNER_IDS


def is_admin_or_owner(member: discord.Member | discord.User) -> bool:
    if isinstance(member, discord.Member):
        if member.guild_permissions.administrator or member.guild_permissions.manage_guild:
            return True
    return is_owner(member.id)


def scope_for_message(message: discord.Message) -> tuple[str, str]:
    if message.guild:
        return "guild", str(message.guild.id)
    return "dm", str(message.author.id)


def guild_prefix(guild: Optional[discord.Guild]) -> str:
    return "Serviço" if guild else "DM"


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc)).strip()


def extract_first_mention(message: discord.Message) -> tuple[Optional[discord.Member], Optional[discord.Role], Optional[discord.abc.GuildChannel]]:
    member = message.mentions[0] if message.mentions else None
    role = message.role_mentions[0] if message.role_mentions else None
    channel = message.channel if isinstance(message.channel, discord.abc.GuildChannel) else None
    return member, role, channel


def parse_channel_ref(text: str, guild: Optional[discord.Guild]) -> Optional[discord.TextChannel | discord.VoiceChannel | discord.StageChannel | discord.Thread]:
    if not guild:
        return None
    t = text.strip()
    m = re.fullmatch(r"<#(\d+)>", t)
    if m:
        ch = guild.get_channel(int(m.group(1)))
        return ch  # type: ignore[return-value]
    if t.isdigit():
        ch = guild.get_channel(int(t))
        return ch  # type: ignore[return-value]
    t = t.lstrip("#").lower()
    for ch in guild.channels:
        if ch.name.lower() == t:
            return ch  # type: ignore[return-value]
    return None


def safe_eval_expr(expr: str) -> Any:
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Load, ast.Tuple,
        ast.List, ast.Dict, ast.Set, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
        ast.GtE, ast.And, ast.Or, ast.BoolOp, ast.Not, ast.Invert, ast.BitAnd, ast.BitOr,
        ast.BitXor, ast.LShift, ast.RShift,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Nó não permitido: {node.__class__.__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, complex, str, bytes, bool, type(None))):
            raise ValueError("Constante não permitida.")
    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})


class ProviderError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, manager: ProviderManager):
        self.manager = manager

    def _provider_config(self, provider: str) -> list[dict[str, Any]]:
        entries = self.manager.entries(provider)
        if not entries:
            raise ProviderError(f"Nenhuma configuração para provider '{provider}'")
        return entries

    async def chat(self, messages: list[dict[str, str]], *, provider: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, timeout: Optional[int] = None) -> str:
        provider_order = self.manager.providers_for(provider)
        last_error: Optional[Exception] = None
        for prov in provider_order:
            try:
                result = await self._chat_provider(prov, messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
                if result:
                    return result
            except Exception as exc:
                last_error = exc
                self.manager.failover(prov)
                continue
        raise ProviderError(f"Falha em todos os providers: {last_error}")

    async def _chat_provider(self, provider: str, messages: list[dict[str, str]], *, temperature: Optional[float], max_tokens: Optional[int], timeout: Optional[int]) -> str:
        entries = self._provider_config(provider)
        entry = self.manager.pick(provider) or entries[0]
        timeout = timeout or int(KEYS_CONFIG.get("defaults", {}).get("timeout_seconds", LLM_TIMEOUT_SECONDS))
        temperature = temperature if temperature is not None else float(KEYS_CONFIG.get("defaults", {}).get("temperature", LLM_TEMPERATURE))
        max_tokens = max_tokens if max_tokens is not None else int(KEYS_CONFIG.get("defaults", {}).get("max_output_tokens", LLM_MAX_OUTPUT_TOKENS))

        if provider == "ollama":
            payload = {"model": entry["model"], "messages": messages, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
            return await asyncio.to_thread(self._ollama_chat, entry["base_url"], payload, timeout)

        if provider in {"openai", "openrouter", "deepseek", "cerebras", "xai"}:
            api_key = entry.get("api_key") or os.getenv(f"{provider.upper()}_API_KEY", "").strip()
            if not api_key:
                raise ProviderError(f"API key ausente para {provider}")
            base_url = (entry.get("base_url") or "").rstrip("/")
            if provider == "xai":
                url = f"{base_url}/chat/completions"
            else:
                url = f"{base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            if provider == "openrouter":
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "https://localhost")
                headers["X-Title"] = os.getenv("OPENROUTER_TITLE", "Omni Admin Bot")
            payload = {
                "model": entry["model"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            return await asyncio.to_thread(self._json_post_chat, url, headers, payload, timeout, provider)

        if provider == "anthropic":
            api_key = entry.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise ProviderError("API key ausente para anthropic")
            url = f"{(entry.get('base_url') or ANTHROPIC_BASE_URL_DEFAULT).rstrip('/')}/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            sys_msg = ""
            msgs: list[dict[str, Any]] = []
            for msg in messages:
                if msg["role"] == "system":
                    sys_msg += msg["content"] + "\n"
                else:
                    role = "assistant" if msg["role"] == "assistant" else "user"
                    msgs.append({"role": role, "content": msg["content"]})
            payload = {
                "model": entry["model"],
                "system": sys_msg.strip(),
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            return await asyncio.to_thread(self._anthropic_chat, url, headers, payload, timeout)

        if provider == "gemini":
            api_key = entry.get("api_key") or os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise ProviderError("API key ausente para gemini")
            model = entry["model"]
            url = f"{(entry.get('base_url') or GEMINI_BASE_URL_DEFAULT).rstrip('/')}/models/{model}:generateContent?key={api_key}"
            payload = self._gemini_payload(messages, temperature, max_tokens)
            return await asyncio.to_thread(self._json_post_gemini, url, payload, timeout)

        raise ProviderError(f"Provider não suportado: {provider}")

    def _json_post_chat(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int, provider: str) -> str:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code >= 400:
            raise ProviderError(f"{provider}: {r.status_code} {r.text[:500]}")
        data = r.json()
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            return msg.get("content") or ""
        return ""

    def _anthropic_chat(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> str:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code >= 400:
            raise ProviderError(f"anthropic: {r.status_code} {r.text[:500]}")
        data = r.json()
        parts = data.get("content") or []
        if parts:
            return "".join(p.get("text", "") for p in parts)
        return ""

    def _gemini_payload(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict[str, Any]:
        system = ""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                system += msg["content"] + "\n"
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system.strip():
            payload["systemInstruction"] = {"parts": [{"text": system.strip()}]}
        return payload

    def _json_post_gemini(self, url: str, payload: dict[str, Any], timeout: int) -> str:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code >= 400:
            raise ProviderError(f"gemini: {r.status_code} {r.text[:500]}")
        data = r.json()
        candidates = data.get("candidates") or []
        if candidates:
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            return "".join(p.get("text", "") for p in parts)
        return ""

    def _ollama_chat(self, base_url: str, payload: dict[str, Any], timeout: int) -> str:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        if r.status_code >= 400:
            raise ProviderError(f"ollama: {r.status_code} {r.text[:500]}")
        data = r.json()
        msg = data.get("message") or {}
        return msg.get("content") or ""


llm = LLMClient(provider_manager)


def build_system_prompt(guild: Optional[discord.Guild]) -> str:
    rules = store.rules()
    rule_lines = "\n".join(f"- {r['name']}: /{r['pattern']}/ -> {r['response']}" for r in rules[:50]) or "(nenhuma)"
    tool_list = "\n".join(f"- {spec.name}: {spec.description}" for spec in registry.list_tools()) or "(nenhuma)"
    guild_name = guild.name if guild else "DM"
    return textwrap.dedent(
        f"""
        Você é um assistente de Discord chamado Omni Admin Bot.
        Trabalhe em português do Brasil, de forma objetiva e útil.
        O usuário prefere interface por tools, não por prefixos dot/slash.
        Quando houver risco, peça confirmação explícita.
        Nunca execute ações destrutivas sem confirmação.
        Se faltar contexto, faça a menor suposição possível.

        Servidor/Canal: {guild_name}

        Tools disponíveis:
        {tool_list}

        Regras customizadas:
        {rule_lines}
        """
    ).strip()


async def maybe_llm_answer(ctx: ToolContext, question: str) -> str:
    if not CHAT_ENABLED:
        return "Chat por LLM está desativado."
    memory = store.recent_memory(ctx.scope, ctx.scope_id, limit=12)
    messages: list[dict[str, str]] = [{"role": "system", "content": build_system_prompt(ctx.guild)}]
    for item in memory:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    reply = await llm.chat(messages)
    return reply.strip() or "Sem resposta do modelo."


def ensure_guild_admin(ctx: ToolContext) -> None:
    author = ctx.author
    if isinstance(author, discord.Member):
        if author.guild_permissions.administrator or author.guild_permissions.manage_guild:
            return
    if is_owner(author.id):
        return
    raise PermissionError("Você não tem permissão para executar esta tool.")


@tool("help", description="Mostra as tools disponíveis e como chamá-las.", category="utility", aliases={"ajuda", "tools", "catalog"})
async def tool_help(ctx: ToolContext) -> str:
    items: list[str] = []
    grouped: dict[str, list[str]] = {}
    for spec in registry.list_tools():
        grouped.setdefault(spec.category, []).append(spec.name)
    for cat in sorted(grouped):
        names = ", ".join(sorted(grouped[cat]))
        items.append(f"**{cat}**: {names}")
    txt_alias_sample = ", ".join(TXT_ALIASES[:12]) if TXT_ALIASES else "(catálogo ausente)"
    return (
        "Use `tool <nome> <args>` ou fale naturalmente com o bot por menção/DM.\n"
        "Ex.: `tool help`, `tool calc 2+2`, `tool remember pizza é preferida`.\n\n"
        + "\n".join(items)
        + f"\n\nAliases legados carregados do TXT: {len(TXT_ALIASES)}.\n"
        + f"Amostra: {txt_alias_sample}"
    )

@tool("ping", description="Mede latência e saúde básica.")
async def tool_ping(ctx: ToolContext) -> str:
    return "pong"

@tool("about", description="Resumo do estado do bot e do catálogo.", aliases={"status", "info"})
async def tool_about(ctx: ToolContext) -> str:
    return (
        f"Bot online. Catálogo TXT: {len(TXT_ALIASES)} aliases. "
        f"Tools registradas: {len(registry.tools)}. "
        f"Legacy prefixes: {'on' if ALLOW_LEGACY_PREFIXES else 'off'}."
    )

@tool("remember", description="Armazena memória curta no contexto atual.", aliases={"memorize"})
async def tool_remember(ctx: ToolContext) -> str:
    content = ctx.args.strip()
    if not content:
        return "Envie o texto após a tool."
    store.add_memory(ctx.scope, ctx.scope_id, "user", content)
    return "Memória salva."

@tool("recall", description="Mostra memórias recentes do contexto.", aliases={"memories"})
async def tool_recall(ctx: ToolContext) -> str:
    rows = store.recent_memory(ctx.scope, ctx.scope_id, limit=10)
    if not rows:
        return "Sem memória salva."
    return "\n".join(f"- {r['role']}: {r['content']}" for r in rows)

@tool("forget", description="Apaga memórias recentes por índice simples ou tudo.", destructive=True)
async def tool_forget(ctx: ToolContext) -> str:
    # Simple but safe: rebuild from scratch only if explicitly asked.
    q = ctx.args.strip().lower()
    if q in {"all", "tudo", "clear", "limpar"}:
        if ctx.scope == "guild":
            scope_clause = ("guild", ctx.scope_id)
        else:
            scope_clause = ("dm", ctx.scope_id)
        store.conn.execute("DELETE FROM memory WHERE scope=? AND scope_id=?", scope_clause)
        store.conn.commit()
        return "Memória apagada."
    return "Use `tool forget all` para limpar tudo."

@tool("tag", description="Cria, lê ou lista tags no escopo atual.")
async def tool_tag(ctx: ToolContext) -> str:
    args = shlex.split(ctx.args) if ctx.args.strip() else []
    if not args:
        return "Use: `tool tag set nome conteúdo` | `tool tag get nome` | `tool tag list`"
    action = args[0].lower()
    scope, scope_id = ctx.scope, ctx.scope_id
    if action == "set" and len(args) >= 3:
        name = args[1].lower()
        content = ctx.args.split(maxsplit=2)[2]
        store.put_tag(scope, scope_id, name, content, str(ctx.author.id))
        return f"Tag `{name}` salva."
    if action == "get" and len(args) >= 2:
        name = args[1].lower()
        content = store.get_tag(scope, scope_id, name)
        return content if content else "Tag não encontrada."
    if action == "list":
        tags = store.list_tags(scope, scope_id)
        return ", ".join(tags) if tags else "Sem tags."
    return "Uso inválido para tag."

@tool("note", description="Adiciona, lista ou remove notas persistentes.")
async def tool_note(ctx: ToolContext) -> str:
    args = shlex.split(ctx.args) if ctx.args.strip() else []
    if not args:
        return "Use: `tool note add texto` | `tool note list` | `tool note delete id`"
    action = args[0].lower()
    scope, scope_id = ctx.scope, ctx.scope_id
    if action == "add" and len(args) >= 2:
        content = ctx.args.split(maxsplit=1)[1]
        note_id = store.add_note(scope, scope_id, str(ctx.author.id), content)
        return f"Nota #{note_id} salva."
    if action == "list":
        notes = store.list_notes(scope, scope_id, limit=10)
        if not notes:
            return "Sem notas."
        return "\n".join(f"#{n['id']} [{n['created_at'][:19]}] {n['content']}" for n in notes)
    if action == "delete" and len(args) >= 2 and args[1].isdigit():
        ok = store.delete_note(int(args[1]))
        return "Nota removida." if ok else "Nota não encontrada."
    return "Uso inválido para note."

@tool("calc", description="Calcula expressões matemáticas com avaliação segura.", aliases={"math"})
async def tool_calc(ctx: ToolContext) -> str:
    expr = ctx.args.strip()
    if not expr:
        return "Envie uma expressão."
    try:
        result = safe_eval_expr(expr)
        return f"{expr} = {result}"
    except Exception as exc:
        return f"Erro no cálculo: {exc}"

@tool("time", description="Mostra a hora atual ou converte fuseaux simples.")
async def tool_time(ctx: ToolContext) -> str:
    q = ctx.args.strip()
    now = utcnow()
    if not q:
        return f"UTC agora: {now.isoformat()}"
    try:
        delta = parse_duration(q)
        if delta:
            target = now + delta
            return f"Em {q}: {target.isoformat()}"
    except Exception:
        pass
    return f"UTC agora: {now.isoformat()}"

@tool("shell", description="Executa um comando local controlado.", destructive=True, admin_only=True)
async def tool_shell(ctx: ToolContext) -> str:
    if not ALLOW_SHELL:
        return "Shell desativado."
    cmd = ctx.args.strip()
    if not cmd:
        return "Envie um comando."
    if blocked_shell_tokens(cmd):
        return "Comando bloqueado por segurança."
    code, out = await run_subprocess(cmd, timeout=20)
    return f"exit={code}\n{safe_short(out, 1500)}"

@tool("search", description="Pesquisa genérica via motor configurável.", aliases={"websearch", "google", "duckduckgo"})
async def tool_search(ctx: ToolContext) -> str:
    q = ctx.args.strip()
    if not q:
        return "Envie um termo de pesquisa."
    endpoint = os.getenv("SEARCH_ENDPOINT", "https://duckduckgo.com/html/")
    params = {"q": q}
    try:
        r = requests.get(endpoint, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = r.text
        titles = re.findall(r'nofollow" class="result__a"[^>]*>(.*?)</a>', text, flags=re.I | re.S)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', text, flags=re.I | re.S)
        results = []
        for i, title in enumerate(titles[:5]):
            clean_title = re.sub(r"<.*?>", "", title)
            snip = re.sub(r"<.*?>", "", snippets[i]) if i < len(snippets) else ""
            results.append(f"- {clean_title}: {snip}")
        return "\n".join(results) if results else "Nenhum resultado extraído."
    except Exception as exc:
        return f"Busca indisponível: {exc}"

@tool("summarize", description="Resume um texto usando LLM.")
async def tool_summarize(ctx: ToolContext) -> str:
    text = ctx.args.strip()
    if not text:
        return "Envie o texto a resumir."
    return await maybe_llm_answer(ctx, f"Resuma o texto de forma objetiva:\n\n{text}")

@tool("rewrite", description="Reescreve um texto em outro tom ou estilo.")
async def tool_rewrite(ctx: ToolContext) -> str:
    text = ctx.args.strip()
    if not text:
        return "Envie o texto."
    return await maybe_llm_answer(ctx, f"Reescreva o texto com clareza e naturalidade:\n\n{text}")

@tool("translate", description="Traduz texto entre idiomas.")
async def tool_translate(ctx: ToolContext) -> str:
    text = ctx.args.strip()
    if not text:
        return "Envie o texto."
    return await maybe_llm_answer(ctx, f"Traduza o texto para o idioma mais provável necessário pelo contexto, e se houver ambiguidade pergunte. Texto:\n\n{text}")

@tool("ask", description="Pergunta livre ao LLM.")
async def tool_ask(ctx: ToolContext) -> str:
    q = ctx.args.strip()
    if not q:
        return "Envie a pergunta."
    return await maybe_llm_answer(ctx, q)

@tool("memory", description="Sinônimo útil para listar memórias.", aliases={"mem", "history"})
async def tool_memory(ctx: ToolContext) -> str:
    return await tool_recall(ctx)

@tool("rules", description="Lista regras de resposta customizadas.")
async def tool_rules(ctx: ToolContext) -> str:
    rules = store.rules()
    if not rules:
        return "Sem regras."
    return "\n".join(f"- {r['name']}: /{r['pattern']}/ => {r['response']}" for r in rules)

@tool("rule", description="Cria ou atualiza uma regra simples.", admin_only=True)
async def tool_rule(ctx: ToolContext) -> str:
    parts = shlex.split(ctx.args) if ctx.args.strip() else []
    if len(parts) < 3:
        return "Use: `tool rule nome padrão resposta...`"
    name, pattern = parts[0], parts[1]
    response = ctx.args.split(maxsplit=2)[2]
    store.add_rule(name, pattern, response, str(ctx.author.id))
    return f"Regra `{name}` salva."

@tool("user", description="Mostra informações do usuário. Aceita menção ou ID.")
async def tool_user(ctx: ToolContext) -> str:
    mention_member, _, _ = extract_first_mention(ctx.message)
    target = mention_member or ctx.author
    if ctx.args.strip().isdigit() and ctx.guild:
        member = ctx.guild.get_member(int(ctx.args.strip()))
        if member:
            target = member
    name = target.display_name if isinstance(target, discord.Member) else target.name
    return f"{name} | id={target.id}"

@tool("channel", description="Mostra informações do canal atual ou de um canal referido.")
async def tool_channel(ctx: ToolContext) -> str:
    if ctx.guild and ctx.args.strip():
        ch = parse_channel_ref(ctx.args.strip(), ctx.guild)
        if ch:
            return f"{ch.name} | id={ch.id} | type={ch.__class__.__name__}"
    ch = ctx.message.channel
    return f"{getattr(ch, 'name', 'dm')} | id={getattr(ch, 'id', 'dm')}"

@tool("guild", description="Mostra informações do servidor atual.")
async def tool_guild(ctx: ToolContext) -> str:
    if not ctx.guild:
        return "Este comando só faz sentido em servidor."
    g = ctx.guild
    return f"{g.name} | id={g.id} | membros={g.member_count or 'desconhecido'}"

@tool("avatar", description="Mostra o avatar de um usuário.")
async def tool_avatar(ctx: ToolContext) -> str:
    target: discord.User | discord.Member = ctx.author
    if ctx.message.mentions:
        target = ctx.message.mentions[0]
    if ctx.args.strip().isdigit() and ctx.guild:
        member = ctx.guild.get_member(int(ctx.args.strip()))
        if member:
            target = member
    return target.display_avatar.url

@tool("purge", description="Remove mensagens em lote.", destructive=True, admin_only=True)
async def tool_purge(ctx: ToolContext) -> str:
    if not isinstance(ctx.channel, discord.TextChannel):
        return "Disponível apenas em canal de texto."
    ensure_guild_admin(ctx)
    n = int(ctx.args.strip() or "0")
    if not (1 <= n <= 200):
        return "Use um número entre 1 e 200."
    deleted = await ctx.channel.purge(limit=n + 1)
    return f"Removidas {max(len(deleted) - 1, 0)} mensagens."

@tool("timeout", description="Aplica timeout em um membro.", destructive=True, admin_only=True)
async def tool_timeout(ctx: ToolContext) -> str:
    if not ctx.guild:
        return "Apenas em servidor."
    ensure_guild_admin(ctx)
    parts = shlex.split(ctx.args) if ctx.args.strip() else []
    if len(parts) < 2 and not ctx.message.mentions:
        return "Use: `tool timeout @membro 10m motivo...`"
    member = ctx.message.mentions[0] if ctx.message.mentions else ctx.guild.get_member(int(parts[0])) if parts[0].isdigit() else None
    if not member:
        return "Membro não encontrado."
    duration = parse_duration(parts[1] if len(parts) > 1 else "10m")
    if not duration:
        return "Duração inválida."
    reason = ctx.args.split(maxsplit=2)[2] if len(parts) >= 3 else None
    until = utcnow() + duration
    await member.timeout(until, reason=reason)
    return f"Timeout aplicado até {until.isoformat()}."

@tool("ban", description="Bane um membro.", destructive=True, admin_only=True)
async def tool_ban(ctx: ToolContext) -> str:
    if not ctx.guild:
        return "Apenas em servidor."
    ensure_guild_admin(ctx)
    member = ctx.message.mentions[0] if ctx.message.mentions else None
    if not member:
        return "Mencione o membro."
    reason = ctx.args.split(maxsplit=1)[1] if " " in ctx.args else None
    await ctx.guild.ban(member, reason=reason, delete_message_days=0)
    return f"{member} banido."

@tool("kick", description="Expulsa um membro.", destructive=True, admin_only=True)
async def tool_kick(ctx: ToolContext) -> str:
    if not ctx.guild:
        return "Apenas em servidor."
    ensure_guild_admin(ctx)
    member = ctx.message.mentions[0] if ctx.message.mentions else None
    if not member:
        return "Mencione o membro."
    reason = ctx.args.split(maxsplit=1)[1] if " " in ctx.args else None
    await ctx.guild.kick(member, reason=reason)
    return f"{member} expulso."

@tool("role", description="Adiciona ou remove cargos.", destructive=True, admin_only=True)
async def tool_role(ctx: ToolContext) -> str:
    if not ctx.guild:
        return "Apenas em servidor."
    ensure_guild_admin(ctx)
    parts = shlex.split(ctx.args) if ctx.args.strip() else []
    if len(parts) < 3:
        return "Use: `tool role add @membro cargo` | `tool role remove @membro cargo`"
    action = parts[0].lower()
    member = ctx.message.mentions[0] if ctx.message.mentions else None
    if not member:
        return "Mencione o membro."
    role_name = ctx.args.split(maxsplit=2)[2]
    role = discord.utils.get(ctx.guild.roles, name=role_name) if ctx.guild else None
    if not role:
        return "Cargo não encontrado."
    if action == "add":
        await member.add_roles(role, reason=f"tool role by {ctx.author}")
        return f"Cargo adicionado a {member.display_name}."
    if action == "remove":
        await member.remove_roles(role, reason=f"tool role by {ctx.author}")
        return f"Cargo removido de {member.display_name}."
    return "Ação inválida."

@tool("lock", description="Tranca o canal atual ou informado.", destructive=True, admin_only=True)
async def tool_lock(ctx: ToolContext) -> str:
    if not ctx.guild:
        return "Apenas em servidor."
    ensure_guild_admin(ctx)
    channel = ctx.message.channel
    if not isinstance(channel, discord.TextChannel):
        return "Apenas em canal de texto."
    overwrite = channel.overwrites_for(ctx.guild.default_role)
    overwrite.send_messages = False
    await channel.set_permissions(ctx.guild.default_role, overwrite=overwrite, reason=f"tool lock by {ctx.author}")
    return f"Canal #{channel.name} trancado."

@tool("unlock", description="Destranca o canal atual.", destructive=True, admin_only=True)
async def tool_unlock(ctx: ToolContext) -> str:
    if not ctx.guild:
        return "Apenas em servidor."
    ensure_guild_admin(ctx)
    channel = ctx.message.channel
    if not isinstance(channel, discord.TextChannel):
        return "Apenas em canal de texto."
    overwrite = channel.overwrites_for(ctx.guild.default_role)
    overwrite.send_messages = None
    await channel.set_permissions(ctx.guild.default_role, overwrite=overwrite, reason=f"tool unlock by {ctx.author}")
    return f"Canal #{channel.name} destrancado."

@tool("set", description="Define uma preferência de escopo.", admin_only=False)
async def tool_set(ctx: ToolContext) -> str:
    parts = shlex.split(ctx.args) if ctx.args.strip() else []
    if len(parts) < 2:
        return "Use: `tool set chave valor`"
    key, value = parts[0], ctx.args.split(maxsplit=1)[1]
    store.set_setting(ctx.scope, ctx.scope_id, key, value)
    return f"{key} salvo."

@tool("get", description="Lê uma preferência de escopo.")
async def tool_get(ctx: ToolContext) -> str:
    key = ctx.args.strip()
    if not key:
        return "Use: `tool get chave`"
    value = store.get_setting(ctx.scope, ctx.scope_id, key)
    return value if value is not None else "Sem valor."

@tool("catalog", description="Lista alguns aliases carregados do TXT.")
async def tool_catalog(ctx: ToolContext) -> str:
    if not TXT_ALIASES:
        return "Catálogo não encontrado."
    return "\n".join(TXT_ALIASES[:80])

@tool("parse", description="Mostra como o bot interpretaria um texto.")
async def tool_parse(ctx: ToolContext) -> str:
    cmd, args = parse_text_command(ctx.args)
    return f"cmd={cmd!r}\nargs={args!r}"

@tool("execute", description="Executa outra tool por nome.", aliases={"call"})
async def tool_execute(ctx: ToolContext) -> str:
    if not ctx.args.strip():
        return "Use: `tool execute nome args...`"
    parts = ctx.args.split(maxsplit=1)
    name = parts[0].lower()
    inner = parts[1] if len(parts) > 1 else ""
    spec = registry.resolve(name)
    if not spec or not spec.handler:
        return "Tool não encontrada."
    nested = dataclasses.replace(ctx, name=spec.name, args=inner)
    result = spec.handler(nested)
    if inspect.isawaitable(result):
        result = await result
    return str(result)

# Synthetic tools based on the TXT catalog for discoverability.
# These do not add fragile prefix commands; they simply give canonical names to known aliases.
for alias in TXT_ALIASES[:]:
    canonical = canonical_from_legacy(alias)
    if registry.resolve(canonical):
        continue
    if canonical.startswith(("allowlist", "blocklist", "loggers", "commands", "ban", "kick", "mute", "unmute", "purge", "timeout")):
        continue

def make_stub(name: str, description: str) -> None:
    if registry.resolve(name):
        return
    @tool(name, description=description, category="legacy")
    async def _stub(ctx: ToolContext, _name=name) -> str:  # type: ignore
        return (
            f"`{_name}` está registrado no catálogo, mas não possui uma implementação local específica.\n"
            f"Use `tool help` para as tools nativas, ou `tool execute ...` para compor outras ações."
        )

def add_catalog_stubs() -> None:
    important = [
        "activity", "allowlist", "blocklist", "commands allowlist", "commands blocklist", "commands usage",
        "avatar", "channel", "guild", "guildicon", "invite", "inviteinfo", "user", "search", "google", "duckduckgo",
        "hash", "convert", "download", "exif", "join", "concat", "extract audio", "extract media", "edit", "interrogate",
        "weather", "time", "help",
    ]
    for name in sorted({canonical_from_legacy(x) for x in TXT_ALIASES}):
        if registry.resolve(name):
            continue
        if any(name == item or name.startswith(item + " ") for item in important):
            continue
        # Only create a moderate number of stubs so help stays sane.
        if len(registry.tools) > 80:
            break
        make_stub(name, "Alias legado do TXT sem binding local.")

add_catalog_stubs()

bot_intents = discord.Intents.default()
bot_intents.message_content = True
bot_intents.guilds = True
bot_intents.members = True
bot_intents.messages = True

class OmniBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(command_prefix=commands.when_mentioned_or(""), intents=bot_intents, help_command=None)
        self.session_started = utcnow()

    async def on_ready(self) -> None:
        print(f"[ready] Logged in as {self.user} ({self.user.id if self.user else 'n/a'})")
        print(f"[ready] tools={len(registry.tools)} aliases={len(TXT_ALIASES)}")
        await self.change_presence(activity=discord.Game(name="tool help"))

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        await self.process_message(message)

    async def process_message(self, message: discord.Message) -> None:
        scope, scope_id = scope_for_message(message)
        content = strip_bot_mention(message) if bot_mentioned(message) else message.content
        command, args = parse_text_command(content)

        if command:
            await self.dispatch_tool(message, command, args, scope, scope_id)
            return

        if bot_mentioned(message) or message.guild is None:
            if content.strip():
                await self.dispatch_tool(message, "ask", content.strip(), scope, scope_id)
                return

    async def dispatch_tool(self, message: discord.Message, command: str, args: str, scope: str, scope_id: str) -> None:
        spec = registry.resolve(command)
        if not spec:
            # Try TXT aliases and whole-message exact match.
            canonical = guess_canonical(command) or guess_canonical(f".{command}") or guess_canonical(f"/{command}")
            if canonical:
                spec = registry.resolve(canonical)
        if not spec and command in {"ask", "chat", "say"}:
            spec = registry.resolve("ask")
        if not spec:
            # Fallback to chat if it looks like plain language.
            if len(command.split()) > 1 or len(args.split()) > 3:
                spec = registry.resolve("ask")
                args = f"{command} {args}".strip()
        if not spec or not spec.handler:
            await message.reply("Tool não reconhecida. Use `tool help`.", mention_author=False)
            return

        ctx = ToolContext(
            bot=self,
            message=message,
            name=spec.name,
            args=args,
            scope=scope,
            scope_id=scope_id,
        )

        if spec.admin_only and not is_admin_or_owner(message.author):
            await message.reply("Você não tem permissão para essa tool.", mention_author=False)
            return

        if spec.destructive and not SAFE_MODE:
            pending = {
                "kind": "tool",
                "guild_id": message.guild.id if message.guild else None,
                "channel_id": message.channel.id,
                "author_id": message.author.id,
                "tool": spec.name,
                "args": args,
            }
            action_id = store.put_pending(pending, DEFAULT_CONF_TIMEOUT)
            await message.reply(
                f"Confirmação necessária para `{spec.name}`.\n"
                f"Responda com `sim {action_id}` para executar ou `não {action_id}` para cancelar.",
                mention_author=False,
            )
            return

        try:
            result = spec.handler(ctx)
            if inspect.isawaitable(result):
                result = await result
            text = str(result)
        except PermissionError as exc:
            text = str(exc)
        except Exception as exc:
            text = f"Erro ao executar `{spec.name}`: {exc}"
        store.audit(
            message.guild.id if message.guild else None,
            message.channel.id,
            message.author.id,
            spec.name,
            {"args": args, "text": message.content},
        )
        store.add_memory(scope, scope_id, "user", message.content)
        if text:
            store.add_memory(scope, scope_id, "assistant", text)
        for chunk in chunk_text(text):
            await message.reply(chunk, mention_author=False)

    async def process_pending_confirmation(self, message: discord.Message) -> bool:
        content = clean_text(message.content).lower()
        parts = content.split(maxsplit=1)
        if not parts:
            return False
        if parts[0] not in {"sim", "não", "nao"}:
            return False
        if len(parts) < 2:
            return False
        action_id = parts[1]
        pending = store.get_pending(action_id)
        if not pending:
            return False
        row = pending.get("_row", {})
        if row.get("author_id") and str(message.author.id) != str(row["author_id"]):
            await message.reply("Esta confirmação não pertence a você.", mention_author=False)
            return True
        if parts[0] in {"não", "nao"}:
            store.delete_pending(action_id)
            await message.reply("Ação cancelada.", mention_author=False)
            return True

        tool_name = pending.get("tool")
        args = pending.get("args", "")
        spec = registry.resolve(tool_name or "")
        if not spec or not spec.handler:
            await message.reply("Ação pendente inválida.", mention_author=False)
            store.delete_pending(action_id)
            return True

        ctx = ToolContext(
            bot=self,
            message=message,
            name=spec.name,
            args=args,
            scope="guild" if message.guild else "dm",
            scope_id=str(message.guild.id if message.guild else message.author.id),
        )
        try:
            result = spec.handler(ctx)
            if inspect.isawaitable(result):
                result = await result
            await message.reply(str(result), mention_author=False)
        except Exception as exc:
            await message.reply(f"Falha ao confirmar: {exc}", mention_author=False)
        finally:
            store.delete_pending(action_id)
        return True

bot = OmniBot()

@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if await bot.process_pending_confirmation(message):
        return
    await bot.process_message(message)

def main() -> None:
    if not TOKEN:
        print("DISCORD_TOKEN ausente.")
        raise SystemExit(2)
    bot.run(TOKEN)

if __name__ == "__main__":
    main()
