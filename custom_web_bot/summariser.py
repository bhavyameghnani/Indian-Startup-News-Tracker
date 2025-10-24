import os
import json
import time
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama     
from langchain_core.messages import SystemMessage, HumanMessage

data_dir = "data_new"
prompt_path = "config/summarization_prompt.txt"

def initialize_llm():
    # Use Gemini via LangChain
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    llm = ChatOllama(model="phi3:latest") 
    return llm

def extract_between_backticks(text: str) -> str:
    """Extracts text between the first set of triple backticks"""
    start = text.find("```")
    if start != -1:
        end = text.find("```", start + 3)
        if end != -1:
            return text[start + 3:end].strip()
    return ""

llm = initialize_llm()
MAX_CONCURRENT_SESSIONS = 5  # Global configuration


# -------------------------
# Summariser Class
# -------------------------

class Summariser:
    def __init__(self, url_conf, prompt_path=prompt_path, data_dir=data_dir):
        self.date = time.strftime("%Y-%m-%d")
        self.id = url_conf["id"]
        self.article_cnt = url_conf.get("article_cnt", 0)

        # Setup dirs
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

        # File paths
        self.new_filepaths = []
        for i in range(self.article_cnt + 1, self.article_cnt + len(new_article_urls) + 1):
            self.new_filepaths.append(os.path.join(self.article_dir, f"{self.id}_{str(i).zfill(4)}.json"))
            # filename = f"{self.id}_{str(i).zfill(4)}.json"
            # self.new_filepaths.append(os.path.join(self.article_dir, filename))

        self.prompt_path = prompt_path
        self.client = llm
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_SESSIONS)

        with open(self.prompt_path, "r", encoding="utf-8") as file:
            self.prompt_template = file.read()

    def generate_summary_message(self, article_text):
        return self.prompt_template.replace("(article)", article_text)

    async def llm_call(self, article_text):
        user_message = self.generate_summary_message(article_text)
        messages = [
        SystemMessage(content="User"),
        HumanMessage(content=user_message)]
        # Gemini via LangChain expects synchronous calls, so use run_in_executor for async
        # messages =[{"role": "user", "content": user_message}]
        chat_response = llm.invoke(messages).content
        # loop = asyncio.get_event_loop()
        print(f"Chat response: {chat_response}")

        s1 = extract_between_backticks(str(chat_response))
        if(s1[0] == "j"): return json.loads(s1[4:])
        return json.loads(s1)



    async def process_article(self, filename):
        print(f"Processing file: {filename}")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"{filename} not found.")
            return

        html_text = data.get("html", "")
        if not html_text.strip():
            print(f"No html text found in {filename}, skipping.")
            return

        if data.get("summary", ""):
            print(f"Summary already exists in {filename}, skipping.")
            return

        async with self.semaphore:
            try:
                result = await self.llm_call(html_text)
                data["summary"] = result.get("summary", "")
                data["title"] = result.get("title", "")
                data["keywords"] = result.get("keywords", [])

                if "html" in data:
                    del data["html"]

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"Summary added and saved to {filename}")
            except Exception as e:
                print(f"Error summarizing {filename}: {e}")

    async def process_all_articles(self):
        print(f"Starting batch processing for {len(self.new_filepaths)} files...")
        tasks = [self.process_article(filepath) for filepath in self.new_filepaths]
        await asyncio.gather(*tasks)
        print("Batch processing completed.")
