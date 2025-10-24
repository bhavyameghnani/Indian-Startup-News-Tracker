import requests
from bs4 import BeautifulSoup
import os
import json
import time


class RssHandler:
    def __init__(self, url_conf, data_dir):
        self.date = time.strftime("%Y-%m-%d")
        self.rss_url = url_conf['rss']
        self.id = url_conf['id']
        self.article_cnt = url_conf.get('article_cnt', 0)

        self.url_dir = os.path.join(data_dir, 'urls')
        self.article_dir = os.path.join(data_dir, 'articles')
        self.output_path = os.path.join(self.url_dir, f"{self.id}.json")

        self.new_url_dir = os.path.join(data_dir, 'new_urls')
        self.new_url_path = os.path.join(self.new_url_dir, f"{self.id}.json")

        os.makedirs(self.new_url_dir, exist_ok=True)
        os.makedirs(self.url_dir, exist_ok=True)
        os.makedirs(self.article_dir, exist_ok=True)

        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w') as f:
                json.dump([], f)

        if not os.path.exists(self.new_url_path):
            with open(self.new_url_path, 'w') as f:
                json.dump([], f)

        self.new_urls = []
        self.new_filepaths = []
        self.urls = []
        self.map = {}

    def extract_urls(self):
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }

            response = requests.get(
                self.rss_url,
                headers=headers,
                timeout=20,
                verify=False,
                proxies=proxies
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to fetch {self.rss_url}: {e}")
            return []

        soup = BeautifulSoup(response.content, 'xml')
        urls = []

        # Try RSS/Atom-style <link> tags inside <item> or <entry>
        for item in soup.find_all(['item', 'entry', 'url']):
            link_tag = item.find("link")
            if link_tag:
                if link_tag.has_attr("href"):
                    urls.append(link_tag['href'])
                elif (
                    link_tag.text.startswith('http')
                    and not link_tag.text.endswith(('png', 'jpg', 'jpeg', 'pdf'))
                ):
                    urls.append(link_tag.text.strip())

            # Handle <loc> tag for sitemaps
            link_tag = item.find('loc')
            if link_tag:
                if link_tag.has_attr('href'):
                    urls.append(link_tag['href'])
                elif (
                    link_tag.text.startswith('http')
                    and not link_tag.text.endswith(('png', 'jpg', 'jpeg', 'pdf'))
                ):
                    urls.append(link_tag.text.strip())

        return urls

    def update_file(self):
        urls = self.extract_urls()
        if not urls:
            print(f"✗ No URLs found at {self.rss_url}")
            return

        with open(self.output_path, "r") as f:
            existing_urls = json.load(f)

        self.new_urls = sorted(set(urls) - set(existing_urls))
        all_urls = list(set(urls) | set(existing_urls))

        with open(self.new_url_path, "w") as f:
            json.dump(self.new_urls, f, indent=2, ensure_ascii=False)

        with open(self.output_path, "w") as f:
            json.dump(all_urls, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(self.new_urls)} new URLs to {self.output_path}")
