import scrapy
from scrapy.crawler import CrawlerProcess
from supabase import create_client
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Supabase configuration
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Scrapy spider definition
class MySpider(scrapy.Spider):
    name = "my_spider"
    start_urls = ["https://example.com"]  # Replace with your target URL

    def parse(self, response):
        # Extract data from the response
        data = {
            "url": response.url,
            "title": response.css("title::text").get(),
            "summary": response.css("meta[name='description']::attr(content)").get(),
            "content": response.css("body").get(),
            "chunk_number": 1,  # Adjust as needed
            "metadata": {"key": "value"},  # Add relevant metadata
            "embedding": [0.1, 0.2, 0.3]  # Replace with actual embedding logic
        }
        
        # Insert data into Supabase
        supabase.table("site_pages").insert(data).execute()

# Function to run the Scrapy spider
def run_spider():
    process = CrawlerProcess()
    process.crawl(MySpider)
    process.start()

# Airflow DAG definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 1),  # Adjust start date
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'scraping_dag',
    default_args=default_args,
    description='A simple web scraping DAG',
    schedule_interval=timedelta(days=1),  # Adjust the schedule as needed
)

# Define the task to run the spider
scrape_task = PythonOperator(
    task_id='run_scrapy_spider',
    python_callable=run_spider,
    dag=dag,
)

scrape_task
