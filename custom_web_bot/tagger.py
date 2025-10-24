import requests
from bs4 import BeautifulSoup
import os
from sentence_transformers import SentenceTransformer, util
import json
import time
import asyncio
from dotenv import load_dotenv
import yaml

load_dotenv()

TAGS_PATH = 'config/tags.json'
MAX_CONCURRENT_SESSIONS = 5  # Global configuration
DATA_DIR = 'data_new'


class Tagger:
    def __init__(self, url_conf, tags_path=TAGS_PATH, data_dir=DATA_DIR):
        self.date = time.strftime("%Y-%m-%d")
        self.id = url_conf['id']
        self.article_cnt = url_conf.get('article_cnt', 0)

        # Directories
        self.new_url_dir = os.path.join(data_dir, 'new_urls')
        os.makedirs(self.new_url_dir, exist_ok=True)

        self.article_dir = os.path.join(data_dir, 'articles')
        os.makedirs(self.article_dir, exist_ok=True)

        # File containing new URLs
        self.new_url_file = os.path.join(self.new_url_dir, f"{self.id}.json")
        if not os.path.exists(self.new_url_file):
            with open(self.new_url_file, 'w') as f:
                json.dump([], f)

        # Load new article URLs
        with open(self.new_url_file, 'r') as f:
            new_article_urls = json.load(f)

        # Map URLs to file paths
        self.new_filepaths = []
        for i in range(self.article_cnt + 1, self.article_cnt + len(new_article_urls) + 1):
            self.new_filepaths.append(
                os.path.join(self.article_dir, f"{self.id}_{str(i).zfill(4)}.json")
            )

        # Load tags config
        with open(tags_path, "r") as f:
            tags = json.load(f)

        self.keys_proper = tags['keywords_proper']
        self.keys_common = tags['keywords_common']

        # Load embedding model (~22MB)
        self.embedding_model = SentenceTransformer(r'config/model/embedding_model')

    def tag_files(self):
        """Process all JSON files and add new tags."""
        json_files = [f for f in self.new_filepaths]
        if not json_files:
            print("No new JSON files.")
            return

        for filename in json_files:
            self.process_single_file(filename)

        print(f"Processed {len(json_files)} files.")

    def bert_embedding_tags(self, article_text, keywords, threshold=0.4):
        article_embedding = self.embedding_model.encode(article_text, convert_to_tensor=True)
        keyword_embeddings = self.embedding_model.encode(keywords, convert_to_tensor=True)

        cosine_scores = util.cos_sim(article_embedding, keyword_embeddings)[0]
        matched_tags = [kw for kw, score in zip(keywords, cosine_scores) if score >= threshold]

        return matched_tags

    def process_single_file(self, file_path):
        """Load JSON, add new keys, and save JSON."""
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            print(f"{file_path}: Not processed")
            return

        summary = str(data.get("summary", ""))
        if not summary:
            return

        print(file_path)
        tags = set()
        subtags = set()

        # 1. Direct string match for proper keywords
        for tag, keywords in self.keys_proper.items():
            for word in keywords:
                if word.lower() in summary.lower():
                    tags.add(tag)
                    subtags.add(word)

        # 2. Semantic match using BERT for common keywords
        for tag, keywords in self.keys_common.items():
            matched_keywords = self.bert_embedding_tags(summary, keywords, threshold=0.3)
            if matched_keywords:
                tags.add(tag)
                subtags.update(matched_keywords)

        # Update data
        data["tags"] = list(tags)
        data["subtags"] = list(subtags)

        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✓ Updated: {os.path.basename(file_path)}")
        return