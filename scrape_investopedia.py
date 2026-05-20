import argparse
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"


def fetch_investopedia_article(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1")

    article = soup.find("article")
    if article is None:
        article = soup.find("div", class_=re.compile(r"(article-body|comp-content|content|article-content)", re.I))

    paragraphs = []
    if article:
        paragraphs = article.find_all("p")
    if not paragraphs:
        paragraphs = soup.find_all("p")

    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    if not text:
        raise ValueError("Could not extract article text from Investopedia page.")

    if title and title.get_text(strip=True):
        text = f"{title.get_text(strip=True)}\n\n{text}"

    return text.strip()


def slugify(url: str) -> str:
    slug = re.sub(r"https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-")
    return slug.lower()


def save_article(text: str, output_dir: Path, url: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"investopedia-{slugify(url)}.txt"
    output_path = output_dir / filename
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Investopedia articles and save text files for RAG ingestion.")
    parser.add_argument("urls", nargs="+", help="Investopedia article URLs to scrape.")
    parser.add_argument("--output-dir", default="data", help="Directory to save scraped article text files.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    for url in args.urls:
        print(f"Scraping: {url}")
        try:
            article_text = fetch_investopedia_article(url)
            saved_path = save_article(article_text, outdir, url)
            print(f"Saved: {saved_path}")
        except Exception as exc:
            print(f"Failed to scrape {url}: {exc}")


if __name__ == "__main__":
    main()
