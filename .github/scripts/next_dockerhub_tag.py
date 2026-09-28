"""Next Docker Hub tag for a fork publish: <upstream version>-dg.<counter>.

The counter is one higher than the highest tag already on Docker Hub for this
project version. A new version starts at 1.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
import urllib.request
from pathlib import Path
from typing import Final

_TAG_PATTERN: Final = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
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


def _hub_token(username: str, password: str) -> str:
    body = json.dumps({"username": username, "password": password}).encode()
    request = urllib.request.Request(
        "https://hub.docker.com/v2/users/login/",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)
    token = payload.get("token")
    if not isinstance(token, str) or token == "":
        raise SystemExit("Docker Hub login did not return a token")
    return token


def tag_names(image: str, token: str | None) -> list[str]:
    names: list[str] = []
    url: str | None = f"https://hub.docker.com/v2/repositories/{image}/tags?page_size=100"
    for _ in range(_PAGE_LIMIT):
        if url is None:
            return names
        request = urllib.request.Request(url)
        if token is not None:
            request.add_header("Authorization", f"JWT {token}")
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        for item in payload.get("results") or []:
            name = item.get("name")
            if isinstance(name, str):
                names.append(name)
        next_url = payload.get("next")
        url = next_url if isinstance(next_url, str) and next_url else None
    raise SystemExit("Docker Hub tag list did not end")


def main() -> None:
    version = project_version(Path("pyproject.toml"))
    image = os.environ.get("IMAGE", "directivegames/litellm")
    username = os.environ.get("DOCKERHUB_USERNAME", "")
    password = os.environ.get("DOCKERHUB_TOKEN", "")
    token = _hub_token(username, password) if username and password else None
    print(next_tag(version, tag_names(image, token)))


if __name__ == "__main__":
    main()
