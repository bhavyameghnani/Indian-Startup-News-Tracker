import os
import json
import sqlite3
import time
from dotenv import load_dotenv

load_dotenv()


class DBHandler:
    def __init__(self, url_conf, db_name, data_dir):
        self.db_name = db_name
        self.date = time.strftime("%Y-%m-%d")
        self.conn = None
        self.cursor = None

        self.id = url_conf["id"]
        self.article_cnt = url_conf.get("article_cnt", 0)

        # Directories
        self.new_url_dir = os.path.join(data_dir, "new_urls")
        os.makedirs(self.new_url_dir, exist_ok=True)

        self.article_dir = os.path.join(data_dir, "articles")
        os.makedirs(self.article_dir, exist_ok=True)

        # New URL file
        self.new_url_file = os.path.join(self.new_url_dir, f"{self.id}.json")
        if not os.path.exists(self.new_url_file):
            with open(self.new_url_file, "w", encoding="utf-8") as f:
                json.dump([], f)

        with open(self.new_url_file, "r", encoding="utf-8") as f:
            new_article_urls = json.load(f)

        # Article file paths
        self.new_filepaths = []
        for i in range(self.article_cnt + 1, self.article_cnt + len(new_article_urls) + 1):
            filename = f"{self.id}_{str(i).zfill(4)}.json"
            self.new_filepaths.append(os.path.join(self.article_dir, filename))

    # ----------------------
    # DB Initialization
    # ----------------------

    def initialize(self):
        # Connect to SQLite database (creates it if not exists)
        self.conn = sqlite3.connect(f"{self.db_name}.db")
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                html TEXT,
                pos TEXT NOT NULL
            );
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS subtags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_tags (
                article_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY(article_id, tag_id),
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            );
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_subtags (
                article_id INTEGER,
                subtag_id INTEGER,
                PRIMARY KEY(article_id, subtag_id),
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                FOREIGN KEY (subtag_id) REFERENCES subtags(id) ON DELETE CASCADE
            );
        """)
        self.conn.commit()

    # ----------------------
    # Insert / Get Helpers
    # ----------------------

    def get_or_create_tag(self, name):
        self.cursor.execute("SELECT id FROM tags WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        self.cursor.execute("INSERT INTO tags (name) VALUES (?)", (name,))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_or_create_subtag(self, name):
        self.cursor.execute("SELECT id FROM subtags WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        self.cursor.execute("INSERT INTO subtags (name) VALUES (?)", (name,))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_article(self, html, pos, tags, subtags):
        self.cursor.execute("INSERT INTO articles (html, pos) VALUES (?, ?)", (html, pos))
        article_id = self.cursor.lastrowid

        for tag in tags:
            tag_id = self.get_or_create_tag(tag)
            self.cursor.execute(
                "INSERT OR IGNORE INTO article_tags (article_id, tag_id) VALUES (?, ?)",
                (article_id, tag_id)
            )

        for subtag in subtags:
            subtag_id = self.get_or_create_subtag(subtag)
            self.cursor.execute(
                "INSERT OR IGNORE INTO article_subtags (article_id, subtag_id) VALUES (?, ?)",
                (article_id, subtag_id)
            )

        self.conn.commit()

    # ----------------------
    # File Processing
    # ----------------------

    def process_file_list(self):
        self.initialize()
        for filename in self.new_filepaths:
            if filename.endswith(".json"):
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    html = data.get("url", "")
                    pos = data.get("path", "")
                    tags = data.get("tags", [])
                    subtags = data.get("subtags", [])

                    if pos:
                        self.insert_article(html, pos, tags, subtags)
                        print(f"Inserted article from {filename}")
                    else:
                        print(f"Skipping {filename}, 'pos' field missing")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        self.close()

    # ----------------------
    # Maintenance
    # ----------------------

    def delete_duplicates(self):
        self.initialize()
        self.cursor.execute("""
            DELETE FROM articles
            WHERE id NOT IN (
                SELECT MIN(id) FROM articles GROUP BY pos, html
            );
        """)
        self.conn.commit()
        print(f"{self.cursor.rowcount} duplicate rows deleted.")
        self.close()

    def delete_date(self, date):
        self.initialize()
        delete_query = "DELETE FROM articles WHERE pos LIKE ?;"
        like_pattern = f"%{date}%"
        self.cursor.execute(delete_query, (like_pattern,))
        self.conn.commit()
        print(f"{self.cursor.rowcount} rows deleted where pos contains '{date}'.")
        self.close()

    def print_all_paths(self):
        self.initialize()
        self.cursor.execute("SELECT pos FROM articles;")
        rows = self.cursor.fetchall()
        print(len(rows), "articles found.")
        for row in rows:
            print(row[0])
        self.close()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


# ----------------------
# Merge Helper
# ----------------------

def merge_files(di, data_dir, map_path):
    id = di["id"]

    new_url_path = os.path.join(data_dir, f"new_urls/{id}.json")
    old_url_path = os.path.join(data_dir, f"urls/{id}.json")

    if not os.path.exists(new_url_path):
        with open(new_url_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(new_url_path, "r", encoding="utf-8") as f:
        new_urls = json.load(f)

    if not os.path.exists(old_url_path):
        with open(old_url_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(old_url_path, "r", encoding="utf-8") as f:
        old_urls = json.load(f)

    all_urls = list(set(new_urls) | set(old_urls))

    with open(new_url_path, "w", encoding="utf-8") as f:
        json.dump(all_urls, f, indent=2)

    new_url_cnt = len(new_urls)
    if new_url_cnt > 0:
        with open(map_path, "r", encoding="utf-8") as f:
            map_data = json.load(f)

        for entry in map_data:
            if entry.get("id") == id:
                entry["article_cnt"] = entry.get("article_cnt", 0) + new_url_cnt
                break

        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(map_data, f, indent=2)
