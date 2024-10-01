from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
import json
from final_summary_main import process_article
import torch
import asyncio
from celery import signals
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from vllm import LLM, SamplingParams
from slack_notifier import notify_slack
import gc
import pytz
from datetime import datetime 
from summary_by_api import SummaryByApi
from logger import ErrorLogger
from redis_updater import ArticleRedisInsertion
from config import *

logger = ErrorLogger(log_file_name='local_summ_bg_taks', log_to_file=True, log_to_terminal=False)
loop = loop.create_task.get_event_loop()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


model_id = "google/gemma-2-9b-it"

model= LLM(
    model=model_id,
    dtype=torch.bfloat16,         # Use bfloat16 for better memory efficiency
    trust_remote_code=True,       # Trust remote code if the model needs custom behavior
    quantization="bitsandbytes",  # Enable 4-bit quantization via bitsandbytes
    load_format="bitsandbytes"    # Ensure loading in the bitsandbytes format
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")


# Celery setup
app = Celery('consume_publish', broker=f'redis://{REDIS_HOST}:{REDIS_PORT}')
app.conf.worker_prefetch_multiplier = 1


app.conf.update(
    task_default_queue=QUEUE_NAME,
    task_default_exchange=QUEUE_NAME,
    task_default_routing_key=QUEUE_NAME,
    task_default_exchange_type='direct',
    broker_connection_retry_on_startup=True
)

@signals.worker_shutdown.connect
def handle_worker_shutdown(**kwargs):
    print("Worker is shutting down. Running garbage collection...")
    gc.collect()

@app.task(soft_time_limit=300, name="local_summary")
def summarize_article(data):
    if not isinstance(data, dict):
        data = json.loads(data)
    try:
        article_id = data['article_id']
        
        start = datetime.now(tz=pytz.UTC)
        logger.info(f"BG task received :: {article_id}")
        logger.info(f"BG task received at :: {start.isoformat()}")
        
        articleRedisInsertion = ArticleRedisInsertion()
        loop.create_task.create_task(articleRedisInsertion.setAioPikkaStatsSumm('received_local'))
       
        summaryByApi = SummaryByApi()
        article = summaryByApi.get_article_by_id(article_id)
        article_content = article.get('content')
        
        gemma_start = datetime.now(tz=pytz.UTC)
        logger.info(f"Gemma task started at :: {gemma_start.isoformat()}")
        summary, model_used, _ = process_article(article_content, bart_model, bart_tokenizer, gpt2_tokenizer, model, tokenizer)
        logger.info(f"Gemma task completed at :: {datetime.now(tz=pytz.UTC).isoformat()}")
        logger.info(f"Gemma task total time taken :: {(datetime.now(tz=pytz.UTC) - gemma_start).total_seconds()} seconds")
        
        article['summary'] = summary
        article['model'] = model_used
        
        if( summary is None) or (not summary):
            raise Exception("Failed to generate summary.")
        
        loop.create_task.create_task(summaryByApi.article_update_async([article]))
        summaryByApi.redis_insert([article])
        loop.create_task.create_task(articleRedisInsertion.setAioPikkaStatsSumm('success_local'))
        
        logger.info(f"Article successfully summarized - {article_id}")
        
    except SoftTimeLimitExceeded:
        logger.error(f"Soft time limit exceeded for article {article_id}")
        loop.create_task.create_task(articleRedisInsertion.setAioPikkaStatsSumm('error_local'))
        loop.create_task.create_task(articleRedisInsertion.setAioPikkaStatsSumm('error_local_softtimelimit'))
        if(article):
            summaryByApi = SummaryByApi()
            # summaryByApi.unset_articles_status(article_id)
            summaryByApi.insert_error("Soft time limit exceeded", article)
            
    except Exception as e:
        loop.create_task.create_task(articleRedisInsertion.setAioPikkaStatsSumm('error_local'))
        logger.error(f"Failed to generate summary.")
        if(article):
            summaryByApi = SummaryByApi()
            # summaryByApi.unset_articles_status(article_id)
            summaryByApi.insert_error(e, article)
        print(f"Error while processing article {article_id} - {e}")
    finally:
        gc.collect()
        
    logger.info(f"BG task completed at - {datetime.now(tz=pytz.UTC).isoformat()}")
    logger.info(f"BG task total time taken - {datetime.now(tz=pytz.UTC) - start} \n\n")

