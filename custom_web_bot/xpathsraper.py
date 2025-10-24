import json
import os
import time
from typing import List, Dict, Any
from urllib.parse import urljoin
from crawl4ai import AsyncWebCrawler
from lxml import html
from crawl4ai.async_configs import CrawlerRunConfig

crawl_config = CrawlerRunConfig(delay_before_return_html=2)
date: str = time.strftime("%Y-%m-%d")
print(date)
dir_path: str = "data_new"

class CustomScraper:
    def __init__(self, config: Dict[str, str], crawl_config: CrawlerRunConfig = crawl_config, dir_path: str = dir_path) -> None:
        """
        Initializes the CustomScraper.

        Parameters:
            - config: Dictionary with keys 'id', 'url', and 'xpath'.
            - crawl_config: Configuration for AsyncwebCrawler.
            - dir_path: Root directory to store output files.
        """
        self.crawler: AsyncWebCrawler = AsyncWebCrawler()
        self.id: str= config["id"]
        self.xpath: str = config["xpath"]
        self.old_url_path: str = os.path.join(dir_path, f"urls/{self.id}.json")
        self.new_url_path: str = os.path.join(dir_path, f"new_urls/{self.id}.json")
        self.crawl_config: CrawlerRunConfig = crawl_config
        self.url: str = config["url"]

    async def fetch_urls(self, url: str, xpath: str) -> List[str]:
        """
        Asynchronously fetch URLs from the given URL and XPath.

        Returns:
            - A list of fully qualified URLs found.
        """
        urls = []
        async with AsyncWebCrawler() as crawler:
            for result in await crawler.arun(url=url):
                if result.markdown:
                    print(result)
                    tree = html.fromstring(result.html)
                    a_tags = tree.xpath(xpath)

                    if not a_tags:
                        print("▲ No links found at given XPath.")
                        return []

                    for a in a_tags:
                        href = a.get("href")
                        if href:
                            full_url = urljoin(url, href)
                            urls.append(full_url)
        return urls

    async def process(self) -> None:
        """
        Main method to process URLs and identify new ones.
        Stores results in JSON files.
        """
        # Ensure old URL file exists
        if not os.path.exists(self.old_url_path):
            os.makedirs(os.path.dirname(self.old_url_path), exist_ok=True)
            with open(self.old_url_path, "w", encoding="utf-8") as f:
                json.dump([], f)

        # Load old URLs
        with open(self.old_url_path, "r", encoding="utf-8") as f:
            try:
                old_urls = set(json.load(f))
            except json.JSONDecodeError:
                old_urls = set()
        
        cur_urls = set(await self.fetch_urls(self.url, self.xpath))
        new_urls = cur_urls - old_urls

        # Always create the new_urls file (even if it's empty)
        os.makedirs(os.path.dirname(self.new_url_path), exist_ok=True)
        with open(self.new_url_path, "w", encoding="utf-8") as f:
            json.dump(list(new_urls), f, indent=2)