from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "deployment" / "compose.weaviate-secure.yaml"
HAPROXY_PATH = PROJECT_ROOT / "deployment" / "haproxy.weaviate-grpc.cfg"


def test_secure_weaviate_compose_is_persistent_authenticated_and_loopback_only() -> None:
    source = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "weaviate:1.39.0" in source
    assert '"127.0.0.1:8080:8080"' in source
    assert '"127.0.0.1:50051:50051"' in source
    assert "rag_weaviate_secure_data" in source
    assert 'AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "false"' in source
    assert 'AUTHENTICATION_APIKEY_ENABLED: "true"' in source
    assert "AUTHENTICATION_APIKEY_ALLOWED_KEYS: ${WEAVIATE_API_KEY:" in source
    assert 'AUTHORIZATION_ADMINLIST_ENABLED: "true"' in source
    assert "AUTOSCHEMA_ENABLED: \"false\"" in source


def test_secure_weaviate_compose_does_not_enable_model_modules() -> None:
    source = COMPOSE_PATH.read_text(encoding="utf-8").lower()

    assert "text2vec" not in source
    assert "generative-" not in source
    assert "reranker-" not in source


def test_secure_weaviate_compose_terminates_grpc_tls_with_http2() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    haproxy = HAPROXY_PATH.read_text(encoding="utf-8")

    assert "haproxy:3.2.23-alpine@sha256:" in compose
    assert '"127.0.0.1:5443:5443"' in compose
    assert "../.local/tailscale-certs:/run/tailscale-certs:ro" in compose
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:" in compose
    assert "bind :5443 ssl" in haproxy
    assert "alpn h2" in haproxy
    assert "timeout client 1h" in haproxy
    assert "timeout server 1h" in haproxy
    assert "timeout client 60s" not in haproxy
    assert "timeout server 60s" not in haproxy
    assert "server weaviate weaviate:50051 check" in haproxy
