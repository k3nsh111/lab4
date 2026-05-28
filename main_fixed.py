import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from urllib.parse import urljoin, urlparse, urldefrag

import aiohttp
import pymorphy3
from bs4 import BeautifulSoup


STOP_WORDS = {
    "и", "в", "во", "на", "с", "к", "ко", "по", "о", "об", "от", "до",
    "за", "из", "у", "а", "но", "или", "что", "как", "это", "его", "ее", "её",
    "их", "для", "при", "над", "под", "не", "же", "бы", "ли", "он", "она", "они",
    "мы", "вы", "я", "ты", "так", "еще", "ещё", "уже", "тоже", "только", "после",
    "перед", "без", "между", "про", "если", "чтобы", "чтоб", "потому", "когда",
    "где", "куда", "который", "которая", "которые", "которое", "один", "два",
    "можно", "нужно", "надо", "будет", "будут", "был", "была", "были", "было",
    "есть", "нет", "да", "все", "всё", "этот", "эта", "эти", "того", "том",
    "тем", "этом", "также", "сейчас", "сегодня", "вчера", "завтра",
    "новость", "сайт", "страница", "читать", "подробно", "главный", "лента",
    "раздел", "рубрика", "фото", "видео", "комментарий", "поделиться",
    "источник", "сообщить", "рассказать", "заявить", "написать", "отметить",
    "ура", "урару", "наш", "ваш", "свой"
}

BAD_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".zip", ".rar",
    ".mp4", ".mp3", ".avi", ".mov", ".css", ".js", ".xml", ".ico"
)

morph = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=50000)
def normalize_word(word: str) -> str:
    return morph.parse(word)[0].normal_form


class Crawler:
    def __init__(self, start_url: str, max_depth: int, workers: int, max_pages: int, output_file: str):
        self.start_url = self.normalize_url(start_url)
        self.domain = self.get_domain(self.start_url)

        self.max_depth = max_depth
        self.workers = workers
        self.max_pages = max_pages
        self.output_file = output_file

        self.queue = asyncio.Queue()
        self.visited = set()
        self.queued = set()
        self.errors = []
        self.status_counter = defaultdict(int)
        self.word_counter = Counter()

        self.pages_downloaded = 0
        self.pages_with_text = 0
        self.total_words = 0
        self.found_links_total = 0

    @staticmethod
    def get_domain(url: str) -> str:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc

    def normalize_url(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        url, _ = urldefrag(url)
        parsed = urlparse(url)
        parsed = parsed._replace(query="")
        return parsed.geturl().rstrip("/")

    def is_internal_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and self.get_domain(url) == self.domain

    @staticmethod
    def is_news_url(url: str) -> bool:
        path = urlparse(url).path.lower()
        return "/news/" in path

    @staticmethod
    def looks_like_file(url: str) -> bool:
        path = urlparse(url).path.lower()
        return path.endswith(BAD_EXTENSIONS)

    async def fetch_html(self, session: aiohttp.ClientSession, url: str) -> str | None:
        try:
            async with session.get(url, allow_redirects=True) as response:
                self.status_counter[response.status] += 1

                if response.status != 200:
                    self.errors.append({"url": url, "status": response.status})
                    return None

                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    return None

                self.pages_downloaded += 1
                return await response.text(errors="ignore")

        except asyncio.TimeoutError:
            self.errors.append({"url": url, "error": "timeout"})
            return None
        except aiohttp.ClientError as error:
            self.errors.append({"url": url, "error": str(error)})
            return None
        except Exception as error:
            self.errors.append({"url": url, "error": f"unexpected error: {error}"})
            return None

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        links = []

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()

            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue

            url = self.normalize_url(urljoin(base_url, href))

            if self.looks_like_file(url):
                continue

            if self.is_internal_url(url) and self.is_news_url(url):
                links.append(url)

        unique_links = list(dict.fromkeys(links))
        self.found_links_total += len(unique_links)
        return unique_links

    @staticmethod
    def extract_article_text(soup: BeautifulSoup) -> str:
        for tag in soup([
            "script", "style", "noscript", "header", "footer",
            "nav", "aside", "form", "button", "svg"
        ]):
            tag.decompose()

        selectors = [
            "article",
            "[itemprop='articleBody']",
            ".article",
            ".article__text",
            ".article-text",
            ".news-text",
            ".publication",
            ".text",
            ".item-text",
            ".article-body"
        ]

        parts = []

        for selector in selectors:
            block = soup.select_one(selector)
            if block:
                parts.append(block.get_text(" ", strip=True))

        if not parts:
            paragraphs = soup.find_all("p")
            parts = [p.get_text(" ", strip=True) for p in paragraphs]

        return " ".join(parts)

    def analyze_text(self, text: str) -> None:
        words = re.findall(r"\b[а-яё]{3,}\b", text.lower())
        useful_words = 0

        for word in words:
            lemma = normalize_word(word)

            if lemma in STOP_WORDS or len(lemma) < 3:
                continue

            self.word_counter[lemma] += 1
            useful_words += 1

        self.total_words += useful_words

        if useful_words > 0:
            self.pages_with_text += 1

    async def process_page(self, session: aiohttp.ClientSession, url: str, depth: int, worker_id: int) -> None:
        print(
            f"[worker {worker_id}] depth={depth} | {url} | "
            f"обработано: {len(self.visited)}/{self.max_pages} | "
            f"в очереди: {self.queue.qsize()} | ошибок: {len(self.errors)}"
        )

        html = await self.fetch_html(session, url)

        if not html:
            return

        soup = BeautifulSoup(html, "lxml")

        text = self.extract_article_text(soup)
        if text:
            self.analyze_text(text)

        if depth >= self.max_depth:
            return

        links = self.extract_links(soup, url)
        added_links = 0

        for link in links:
            if len(self.queued) >= self.max_pages:
                break

            if link not in self.visited and link not in self.queued:
                self.queued.add(link)
                await self.queue.put((link, depth + 1))
                added_links += 1

        print(
            f"[worker {worker_id}] найдено ссылок: {len(links)} | "
            f"добавлено в очередь: {added_links} | всего запланировано: {len(self.queued)}"
        )

    async def worker(self, session: aiohttp.ClientSession, worker_id: int) -> None:
        while True:
            try:
                url, depth = await self.queue.get()
            except asyncio.CancelledError:
                return

            try:
                if url not in self.visited and len(self.visited) < self.max_pages:
                    self.visited.add(url)
                    await self.process_page(session, url, depth, worker_id)
            except Exception as error:
                self.errors.append({"url": url, "error": f"worker error: {error}"})
            finally:
                self.queue.task_done()

    async def run(self) -> None:
        self.queued.add(self.start_url)
        await self.queue.put((self.start_url, 0))

        timeout = aiohttp.ClientTimeout(total=10, connect=4, sock_read=6)

        connector = aiohttp.TCPConnector(
            limit=self.workers,
            limit_per_host=self.workers
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36 EducationalCrawler/1.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"
        }

        async with aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=connector
        ) as session:
            tasks = [
                asyncio.create_task(self.worker(session, i + 1))
                for i in range(self.workers)
            ]

            await self.queue.join()

            for task in tasks:
                task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

        result = {
            "start_url": self.start_url,
            "domain": self.domain,
            "max_depth": self.max_depth,
            "workers": self.workers,
            "max_pages": self.max_pages,
            "pages_processed": len(self.visited),
            "pages_downloaded": self.pages_downloaded,
            "pages_with_text": self.pages_with_text,
            "total_useful_words": self.total_words,
            "found_links_total": self.found_links_total,
            "http_statuses": dict(sorted(self.status_counter.items())),
            "top_10_words": self.word_counter.most_common(10),
            "errors_count": len(self.errors),
            "errors_sample": self.errors[:20]
        }

        with open(self.output_file, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

        print(f"\nГотово. Результат сохранен в {self.output_file}")
        print("\nТоп-10 слов:")

        for word, count in result["top_10_words"]:
            print(f"{word}: {count}")

        print(f"\nОбработано страниц: {len(self.visited)}")
        print(f"Успешно скачано HTML-страниц: {self.pages_downloaded}")
        print(f"Страниц с найденным текстом: {self.pages_with_text}")
        print(f"Ошибок: {len(self.errors)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Асинхронный crawler для сайта новостей URA.NEWS")
    parser.add_argument("url", help="Стартовый URL, например: https://ura.news/")
    parser.add_argument("--depth", type=int, default=2, help="Максимальная глубина обхода")
    parser.add_argument("--workers", type=int, default=5, help="Количество асинхронных обработчиков")
    parser.add_argument("--max-pages", type=int, default=30, help="Максимальное количество страниц")
    parser.add_argument("--output", default="result.json", help="Файл для сохранения результата")

    args = parser.parse_args()

    if args.depth < 0:
        parser.error("--depth не может быть меньше 0")
    if args.workers < 1:
        parser.error("--workers должен быть не меньше 1")
    if args.max_pages < 1:
        parser.error("--max-pages должен быть не меньше 1")

    return args


async def main() -> None:
    args = parse_args()

    crawler = Crawler(
        start_url=args.url,
        max_depth=args.depth,
        workers=args.workers,
        max_pages=args.max_pages,
        output_file=args.output
    )

    await crawler.run()


if __name__ == "__main__":
    asyncio.run(main())
