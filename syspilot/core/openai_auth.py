from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
SCOPE = "openid profile email offline_access"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
ACCOUNT_CLAIM = "https://api.openai.com/auth"
SUPPORTED_CODEX_MODELS = ("gpt-5.2-codex", "gpt-5.3-codex")


def syspilot_openai_auth_file() -> Path:
    return Path("~/.syspilot/auth/openai.json").expanduser()


def opencode_openai_auth_file() -> Path:
    return Path("~/.opencode/auth/openai.json").expanduser()


def create_pkce_verifier_and_challenge() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("utf-8").rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest())
        .decode("utf-8")
        .rstrip("=")
    )
    return verifier, challenge


def create_state() -> str:
    return secrets.token_hex(16)


def build_authorization_url(state: str, challenge: str) -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "codex_cli_rs",
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    raw = (value or "").strip()
    if not raw:
        return None, None

    try:
        parsed = urlparse(raw)
        code = None
        state = None
        if parsed.query:
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            code = query.get("code")
            state = query.get("state")
            if code:
                return code, state
    except Exception:
        pass

    if "#" in raw:
        code, state = raw.split("#", 1)
        return code or None, state or None

    if "code=" in raw:
        query = dict(parse_qsl(raw, keep_blank_values=True))
        return query.get("code"), query.get("state")

    return raw, None


def wait_for_callback_code(
    state: str,
    timeout_seconds: int,
    authorize_url: str | None = None,
) -> tuple[str | None, bool]:
    result: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            req_state = query.get("state")
            code = query.get("code")

            if req_state != state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State mismatch")
                return

            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing authorization code")
                return

            result["code"] = code
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authentication successful. You can close this window.")

        def log_message(self, format: str, *args: object) -> None:
            return

    try:
        server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
    except OSError:
        return None, False

    browser_opened = open_browser(authorize_url) if authorize_url else False
    if authorize_url and not browser_opened:
        server.server_close()
        return None, False

    deadline = time.time() + max(timeout_seconds, 1)
    server.timeout = 0.25
    try:
        while time.time() < deadline and "code" not in result:
            server.handle_request()
    finally:
        server.server_close()

    return result.get("code"), browser_opened


def open_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception:
        return False


def exchange_authorization_code(code: str, verifier: str) -> dict[str, str | int]:
    response = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )

    if response.status_code >= 400:
        raise ValueError(
            f"Failed to exchange OpenAI authorization code: {response.status_code} {response.text}"
        )

    payload = response.json()
    access = payload.get("access_token")
    refresh = payload.get("refresh_token")
    expires_in_raw = payload.get("expires_in")

    try:
        expires_in = int(expires_in_raw)
    except (TypeError, ValueError):
        expires_in = None

    if not access or not refresh or expires_in is None:
        raise ValueError("OpenAI token response missing required fields")

    return {
        "type": "oauth",
        "access": str(access),
        "refresh": str(refresh),
        "expires": int(time.time() * 1000) + expires_in * 1000,
    }


def save_auth(auth: dict[str, str | int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(auth, f, indent=2)
    path.chmod(0o600)
