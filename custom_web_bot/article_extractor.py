import os
import json
import time
import asyncio
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, CacheMode, RateLimiter
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.async_configs import CrawlerRunConfig, BrowserConfig

# Load environment variables
load_dotenv()

# Rate limiter
rate_limiter = RateLimiter(
    base_delay=(2.0, 4.0),  # random delay between 2–4 seconds
    max_delay=30.0,         # cap delay at 30 seconds
    max_retries=5,          # retry up to 5 times
    rate_limit_codes=[429, 503]  # handle HTTP status codes
)

# Dispatcher
dispatcher = MemoryAdaptiveDispatcher(
    memory_threshold_percent=80.0,
    check_interval=1.0,
    max_session_permit=5,
    rate_limiter=RateLimiter(base_delay=(1.0, 2.0), max_delay=6.0, max_retries=2)
)

# Crawler configuration
crawler_conf = CrawlerRunConfig(
    magic=True,
    simulate_user=True,
    override_navigator=True,
    stream=True,
)

# Paths
data_dir = "data_new"
prompt_path = "summarization_prompt.txt"


class ArticleExtractor:
    def __init__(self, url_conf, data_dir=data_dir, browser_conf=None, dispatcher=dispatcher, crawler_conf=crawler_conf):
        self.date = time.strftime("%Y-%m-%d")
        self.id = url_conf["id"]
        self.article_cnt = url_conf.get("article_cnt", 0)

        self.new_url_path = os.path.join(data_dir, f"new_urls/{self.id}.json")
        self.article_dir = os.path.join(data_dir, "articles")
        os.makedirs(self.article_dir, exist_ok=True)

        self.crawler_conf = crawler_conf
        self.dispatcher = dispatcher
        self.browser_conf = browser_conf

        self.new_urls = []
        if os.path.exists(self.new_url_path):
            with open(self.new_url_path, "r", encoding="utf-8") as f:
                self.new_urls = json.load(f)

        self.map = {}
        self.map_urls()

    def map_urls(self):
        for i, url in enumerate(self.new_urls, start=1):
            self.map[url] = self.article_cnt + i
        self.article_cnt += len(self.new_urls)

    def save_json(self, url, html, fileid):
        data = {
            "url": url,
            "html": html,
            "path": fileid,
            "title": "",
            "date": self.date,
        }
        filename = os.path.join(self.article_dir, f"{fileid}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to {filename}")

    def process_result(self, result):
        try:
            url = result.url
            html = result.markdown or result.html or ""

            i = self.map[url]
            fileid = f"{self.id}_{str(i).zfill(4)}"

            self.save_json(url, html, fileid)
        except Exception as e:
            print(f"Error processing {getattr(result, 'url', 'unknown')}: {e}")

    async def extract_articles(self):
        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun_many(urls=self.new_urls)
            for result in results:
                if result.success:
                    self.process_result(result)
