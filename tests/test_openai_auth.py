from urllib.parse import parse_qs, urlparse

from syspilot.core.openai_auth import (
    AUTHORIZE_URL,
    CLIENT_ID,
    REDIRECT_URI,
    SCOPE,
    build_authorization_url,
    create_pkce_verifier_and_challenge,
    create_state,
    parse_authorization_input,
    syspilot_openai_auth_file,
)


def test_build_authorization_url_contains_required_query_parameters():
    verifier, challenge = create_pkce_verifier_and_challenge()
    assert verifier
    assert challenge

    state = create_state()
    url = build_authorization_url(state, challenge)
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == AUTHORIZE_URL
    assert params["response_type"] == ["code"]
    assert params["client_id"] == [CLIENT_ID]
    assert params["redirect_uri"] == [REDIRECT_URI]
    assert params["scope"] == [SCOPE]
    assert params["code_challenge"] == [challenge]
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"] == [state]


def test_parse_authorization_input_from_full_redirect_url():
    code, state = parse_authorization_input(
        "http://localhost:1455/auth/callback?code=abc123&state=state456"
    )
    assert code == "abc123"
    assert state == "state456"


def test_parse_authorization_input_from_compact_format():
    code, state = parse_authorization_input("abc123#state456")
    assert code == "abc123"
    assert state == "state456"


def test_default_auth_path_points_to_syspilot_directory():
    assert str(syspilot_openai_auth_file()).endswith("/.syspilot/auth/openai.json")
