"""Tests for the extra CA trust hook."""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import os
import ssl
from collections.abc import Iterator
from pathlib import Path

import aiohttp
import certifi
import httpx
import pytest
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from music_assistant.helpers import custom_ca
from music_assistant.helpers import ssl as mass_ssl

_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", custom_ca.ENV_EXTRA_CA_FILE)


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _make_ca_and_leaf(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a throwaway CA and a localhost leaf signed by it (ca.pem, leaf.pem, leaf.key)."""
    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("Test Private CA"))
        .issuer_name(_name("Test Private CA"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("localhost"))
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=2))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_pem = tmp_path / "ca.pem"
    ca_pem.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    leaf_pem = tmp_path / "leaf.pem"
    leaf_pem.write_bytes(leaf_cert.public_bytes(serialization.Encoding.PEM))
    leaf_key_pem = tmp_path / "leaf.key"
    leaf_key_pem.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return ca_pem, leaf_pem, leaf_key_pem


@pytest.fixture(autouse=True)
def restore_process_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo the env and ssl patches the hook makes."""
    original = ssl.create_default_context
    original_where, original_contents = certifi.where, certifi.contents
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    ssl.create_default_context = original
    certifi.where, certifi.contents = original_where, original_contents
    for var in _ENV_VARS:
        os.environ.pop(var, None)


async def _fetch_both(port: int) -> tuple[bool, bool]:
    """Return whether an aiohttp and an httpx client can verify the local server."""
    url = f"https://localhost:{port}/"
    try:
        async with aiohttp.ClientSession() as session, session.get(url) as resp:
            aiohttp_ok = resp.status == 200
    except aiohttp.ClientConnectorCertificateError:
        aiohttp_ok = False
    try:
        async with httpx.AsyncClient() as client:
            httpx_ok = (await client.get(url)).status_code == 200
    except httpx.ConnectError:
        httpx_ok = False
    return aiohttp_ok, httpx_ok


async def _run_with_server(
    leaf: Path, key: Path, ca_dir: Path, data_dir: Path
) -> tuple[tuple[bool, bool], tuple[bool, bool]]:
    app = web.Application()

    async def handler(_: web.Request) -> web.Response:
        return web.Response(text="ok")

    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(leaf), str(key))
    site = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=server_ctx)
    await site.start()
    port = runner.addresses[0][1]
    try:
        before = await _fetch_both(port)
        custom_ca.setup_custom_ca_trust(data_dir, (ca_dir,))
        return before, await _fetch_both(port)
    finally:
        await runner.cleanup()


def test_extra_ca_makes_clients_verify(tmp_path: Path) -> None:
    """Aiohttp and httpx fail to verify before the hook and succeed after it."""
    ca, leaf, key = _make_ca_and_leaf(tmp_path)
    ca_dir = tmp_path / "cas"
    ca_dir.mkdir()
    (ca_dir / "private.crt").write_bytes(ca.read_bytes())
    before, after = asyncio.run(_run_with_server(leaf, key, ca_dir, tmp_path / "data"))
    assert before == (False, False)
    assert after == (True, True)


def test_env_var_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MASS_EXTRA_CA_FILE is honoured and exports the bundle env vars."""
    ca, _leaf, _key = _make_ca_and_leaf(tmp_path)
    monkeypatch.setenv(custom_ca.ENV_EXTRA_CA_FILE, str(ca))
    bundle = custom_ca.setup_custom_ca_trust(tmp_path / "data", (tmp_path / "missing",))
    assert bundle is not None
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        assert Path(os.environ[var]) == bundle
    assert "BEGIN CERTIFICATE" in bundle.read_text()


def test_no_file_is_noop(tmp_path: Path) -> None:
    """Without any CA file nothing changes and verification is not weakened."""
    original = ssl.create_default_context
    assert custom_ca.setup_custom_ca_trust(tmp_path, (tmp_path / "none",)) is None
    assert ssl.create_default_context is original
    assert "SSL_CERT_FILE" not in os.environ
    assert ssl.create_default_context().verify_mode == ssl.CERT_REQUIRED


def test_garbage_file_ignored(tmp_path: Path) -> None:
    """A file without certificates adds nothing."""
    ca_dir = tmp_path / "cas"
    ca_dir.mkdir()
    (ca_dir / "bad.pem").write_text("not a certificate")
    assert custom_ca.setup_custom_ca_trust(tmp_path / "data", (ca_dir,)) is None


async def _ws_ok(port: int) -> bool:
    """Open a websocket the way the HA client does: on a session with Music Assistant's context."""
    connector = aiohttp.TCPConnector(ssl=mass_ssl.client_context())
    try:
        async with (
            aiohttp.ClientSession(connector=connector) as session,
            session.ws_connect(f"wss://localhost:{port}/ws") as ws,
        ):
            await ws.send_str("ping")
            return (await ws.receive_str()) == "ping"
    except aiohttp.ClientConnectorCertificateError:
        return False


async def _run_ws(leaf: Path, key: Path, ca_dir: Path, data_dir: Path) -> tuple[bool, bool]:
    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(await ws.receive_str())
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(leaf), str(key))
    await web.TCPSite(runner, "127.0.0.1", 0, ssl_context=server_ctx).start()
    port = runner.addresses[0][1]
    try:
        before = await _ws_ok(port)
        custom_ca.setup_custom_ca_trust(data_dir, (ca_dir,))
        return before, await _ws_ok(port)
    finally:
        await runner.cleanup()


def test_websocket_on_mass_ssl_context(tmp_path: Path) -> None:
    """The import-time contexts of music_assistant.helpers.ssl also trust the extra CA."""
    ca, leaf, key = _make_ca_and_leaf(tmp_path)
    ca_dir = tmp_path / "cas"
    ca_dir.mkdir()
    (ca_dir / "private.crt").write_bytes(ca.read_bytes())
    assert asyncio.run(_run_ws(leaf, key, ca_dir, tmp_path / "data")) == (False, True)
