"""
Trust extra (private) certificate authorities.

Some setups run services (for example Home Assistant) behind a certificate that is signed by a
private CA. A server side client (like the Home Assistant login flow) can only verify such a
certificate when that CA is in its trust store. Container based installs can not edit the system
store persistently, so this module builds a combined bundle at startup from the system CAs plus
the CA files found in a mounted folder and points the usual TLS libraries at it.

Verification is never disabled: the extra CAs are only added to the set of trusted roots.
"""

from __future__ import annotations

import logging
import os
import ssl
import sys
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.x509.oid import NameOID

from music_assistant.constants import MASS_LOGGER_NAME

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.custom_ca")

ENV_EXTRA_CA_FILE = "MASS_EXTRA_CA_FILE"
DEFAULT_CA_DIRS = (
    Path("/ssl/mass-trusted-cas"),
    Path("/share/mass-trusted-cas"),
    Path("/media/mass-trusted-cas"),
)
CA_FILE_SUFFIXES = (".pem", ".crt")
BUNDLE_FILENAME = "custom-ca-bundle.pem"
EXTRA_ONLY_FILENAME = "custom-ca-extra.pem"
_PEM_BEGIN = "-----BEGIN CERTIFICATE-----"
_PEM_END = "-----END CERTIFICATE-----"


def find_extra_ca_files(search_dirs: tuple[Path, ...] = DEFAULT_CA_DIRS) -> list[Path]:
    """Return the extra CA files: the MASS_EXTRA_CA_FILE env var, then the folders' files."""
    found: list[Path] = []
    if (env_file := os.environ.get(ENV_EXTRA_CA_FILE)) and Path(env_file).is_file():
        found.append(Path(env_file))
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        found.extend(
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in CA_FILE_SUFFIXES
        )
    return found


def _read_certificates(path: Path) -> list[str]:
    """Read the PEM certificates of a file (PEM blocks only, anything else is ignored)."""
    text = path.read_text(encoding="ascii", errors="ignore")
    pems: list[str] = []
    while (start := text.find(_PEM_BEGIN)) != -1:
        end = text.find(_PEM_END, start)
        if end == -1:
            break
        end += len(_PEM_END)
        pems.append(text[start:end])
        text = text[end:]
    return pems


def _subject_name(pem: str) -> str:
    """Return a printable subject (CN, else the full RFC4514 string) of a PEM certificate."""
    cert = x509.load_pem_x509_certificate(pem.encode("ascii"))
    if cn := cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME):
        return str(cn[0].value)
    return cert.subject.rfc4514_string()


def _base_bundle() -> str:
    """Return the PEM text of the CA bundle OpenSSL/Python (and certifi) would use."""
    parts: list[str] = []
    paths = ssl.get_default_verify_paths()
    candidates = [paths.cafile, paths.openssl_cafile]
    try:
        import certifi  # noqa: PLC0415

        candidates.append(certifi.where())
    except ImportError:
        pass
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen and Path(candidate).is_file():
            seen.add(candidate)
            parts.append(Path(candidate).read_text(encoding="ascii", errors="ignore"))
    return "\n".join(parts)


def _patch_default_ssl_context(extra_path: Path) -> None:
    """
    Add the extra CAs to default client contexts.

    SSL_CERT_FILE is ignored by Python on Windows and by libraries that set their own CA file,
    so a default SERVER_AUTH context (no explicit CA arguments) also loads the extra CAs.
    """
    original = ssl.create_default_context

    @wraps(original)
    def create_default_context(
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH, *args: Any, **kwargs: Any
    ) -> ssl.SSLContext:
        context = original(purpose, *args, **kwargs)
        if purpose == ssl.Purpose.SERVER_AUTH:
            context.load_verify_locations(cafile=str(extra_path))
        return context

    ssl.create_default_context = create_default_context
    # aiohttp builds its verified context once, at import time: add the CAs to that one too
    if (connector := sys.modules.get("aiohttp.connector")) is not None:
        connector._SSL_CONTEXT_VERIFIED.load_verify_locations(cafile=str(extra_path))
    # so does Music Assistant's own ssl helper (used by the shared http_session and so by the
    # Home Assistant websocket client)
    if (mass_ssl := sys.modules.get("music_assistant.helpers.ssl")) is not None:
        for context in (mass_ssl._DEFAULT_SSL_CONTEXT, *mass_ssl._SSL_CONTEXTS.values()):
            context.load_verify_locations(cafile=str(extra_path))


def setup_custom_ca_trust(
    data_dir: str | Path | None = None,
    search_dirs: tuple[Path, ...] = DEFAULT_CA_DIRS,
) -> Path | None:
    """
    Make the process trust the extra CAs found on disk. A no-op when there are none.

    Call this at process start, before any HTTP client is created.

    :param data_dir: Writable folder for the combined bundle (default: a temp folder).
    :param search_dirs: Folders searched for *.pem / *.crt files.
    :return: The combined bundle path, or None when nothing was added.
    """
    extra_pems: list[str] = []
    names: list[str] = []
    for ca_file in find_extra_ca_files(search_dirs):
        try:
            for pem in _read_certificates(ca_file):
                names.append(_subject_name(pem))
                extra_pems.append(pem)
        except (OSError, ValueError) as err:
            LOGGER.warning("Ignoring unusable extra CA file %s: %s", ca_file, err)
    if not extra_pems:
        return None

    target_dir = Path(data_dir) if data_dir else Path(tempfile.gettempdir())
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = target_dir / BUNDLE_FILENAME
        extra_path = target_dir / EXTRA_ONLY_FILENAME
        extra_path.write_text("\n".join(extra_pems) + "\n", encoding="ascii")
        bundle_path.write_text(
            _base_bundle() + "\n" + "\n".join(extra_pems) + "\n", encoding="ascii"
        )
    except OSError as err:
        LOGGER.warning("Could not write the extra CA bundle, trust unchanged: %s", err)
        return None

    for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ[env_var] = str(bundle_path)
    # libraries that call certifi.where() directly (instead of the default OpenSSL paths)
    try:
        import certifi  # noqa: PLC0415

        certifi.where = lambda: str(bundle_path)
        certifi.contents = lambda: bundle_path.read_text(encoding="ascii")
    except ImportError:
        pass
    _patch_default_ssl_context(extra_path)
    LOGGER.info("Added %d extra trusted CA(s): %s", len(names), ", ".join(names))
    return bundle_path
