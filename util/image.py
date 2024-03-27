import httpx


def download_file(image_url: str) -> bytes:
    r = httpx.get(image_url, timeout=1000)
    return r.content
