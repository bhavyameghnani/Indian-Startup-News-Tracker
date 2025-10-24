from url_extractor import RssHandler
from article_extractor import ArticleExtractor
from summariser import Summariser
from tagger import Tagger
from dbmanager import DBHandler, merge_files
from xpathsraper import CustomScraper
import asyncio
import os
import json

data_dir = 'data_new'
tags_path = 'config/tags.json'
map_path = "sample_mapping.json"

with open(map_path, 'r') as f:
    maps = json.load(f)

for conf in maps:

    if(conf["type"] == "rss"):
        handler = RssHandler(conf, data_dir=data_dir)
        handler.update_file()
    elif(conf["type"] == "xpath"):
        sc = CustomScraper(config=conf)
        asyncio.run(sc.process())

    extractor = ArticleExtractor(url_conf=conf, data_dir=data_dir)
    asyncio.run(extractor.extract_articles())

    summariser = Summariser(conf)
    asyncio.run(summariser.process_all_articles())

    tt = Tagger(url_conf=conf, tags_path=tags_path)
    tt.tag_files()

    agent = DBHandler(url_conf=conf, db_name="news", data_dir='data_new')
    agent.process_file_list()
    agent.delete_duplicates()

    merge_files(conf, data_dir, map_path)