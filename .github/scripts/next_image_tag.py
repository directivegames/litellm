"""Next registry tag for a fork publish: <upstream version>-dg.<counter>.

The counter is one higher than the highest tag already in the registry for this
project version. A new version starts at 1. The image lives on GitHub Container
Registry, and its tags are read through the standard registry API.

Callers look for: next_tag, tag_names, main.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Final

_TAG_PATTERN: Final = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_NEXT_LINK: Final = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
_PAGE_LIMIT: Final = 50


def project_version(path: Path) -> str:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or version == "":
        raise SystemExit(f"{path} has no project.version")
    return version


def next_tag(version: str, existing: list[str]) -> str:
    """Return version-dg.N. N is 1 when no version-dg.N tag exists yet."""
    pattern = re.compile(rf"^{re.escape(version)}-dg\.([0-9]+)$")
    highest = 0
    for name in existing:
        match = pattern.fullmatch(name)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    tag = f"{version}-dg.{highest + 1}"
    if _TAG_PATTERN.fullmatch(tag) is None:
        raise SystemExit(f"{tag} is not a valid Docker tag")
    return tag


def split_image(image: str) -> tuple[str, str]:
    """Split ghcr.io/owner/name into the registry host and the repository path."""
    host, _, repository = image.partition("/")
    if "." not in host or repository == "":
        raise SystemExit(f"{image} does not name a registry host and repository")
    return host, repository


def _registry_token(host: str, repository: str, username: str, password: str) -> str:
    query = urllib.parse.urlencode({"scope": f"repository:{repository}:pull", "service": host})
    request = urllib.request.Request(f"https://{host}/token?{query}")
    if username and password:
        basic = base64.b64encode(f"{username}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {basic}")
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)
    token = payload.get("token") or payload.get("access_token")
    if not isinstance(token, str) or token == "":
        raise SystemExit(f"{host} did not return a registry token")
    return token


def tag_names(host: str, repository: str, token: str) -> list[str]:
    names: list[str] = []
    url: str | None = f"https://{host}/v2/{repository}/tags/list?n=1000"
    for _ in range(_PAGE_LIMIT):
        if url is None:
            return names
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(request) as response:
                payload = json.load(response)
                link = response.headers.get("Link", "")
        except urllib.error.HTTPError as error:
            # Only "no such package" means no tags yet. A refused login must fail,
            # or the counter would restart at 1 and overwrite a published tag.
            if error.code == 404 and not names:
                return names
            raise
        for name in payload.get("tags") or []:
            if isinstance(name, str):
                names.append(name)
        match = _NEXT_LINK.search(link)
        url = urllib.parse.urljoin(f"https://{host}", match.group(1)) if match else None
    raise SystemExit("Registry tag list did not end")


def main() -> None:
    version = project_version(Path("pyproject.toml"))
    host, repository = split_image(os.environ.get("IMAGE", "ghcr.io/directivegames/litellm"))
    username = os.environ.get("REGISTRY_USERNAME", "")
    password = os.environ.get("REGISTRY_TOKEN", "")
    token = _registry_token(host, repository, username, password)
    print(next_tag(version, tag_names(host, repository, token)))


if __name__ == "__main__":
    main()
