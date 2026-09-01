from __future__ import annotations

import argparse
from pathlib import Path

import requests


def download(
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    print(f"Downloading:")
    print(f"  {url}")
    print(f"→ {destination}")

    with requests.get(
        url,
        stream=True,
        timeout=(30, 300),
        headers={
            "User-Agent": "kmz2geolibre/1.0",
        },
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get(
                "Content-Length",
                "0",
            )
        )

        downloaded = 0

        with temporary.open("wb") as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = (
                        downloaded / total * 100
                    )

                    print(
                        f"\r"
                        f"{downloaded / 1024 / 1024:,.1f} "
                        f"/ "
                        f"{total / 1024 / 1024:,.1f} MB "
                        f"({percent:5.1f}%)",
                        end="",
                    )

    print()

    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "url",
    )

    parser.add_argument(
        "destination",
        type=Path,
    )

    args = parser.parse_args()

    download(
        args.url,
        args.destination,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())