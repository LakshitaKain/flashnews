from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.routing import APIRoute
from transformers import GPT2TokenizerFast
from transformers import BartTokenizer
import uvicorn
from slack_notifier import notify_slack
from fastapi.responses import JSONResponse 
from langchain.text_splitter import TokenTextSplitter
import traceback
import time
import sys
import requests
from uuid import uuid4
import json
import re
import torch
from datetime import datetime
import warnings
from transformers import BartTokenizer, BartForConditionalGeneration
from database_connection import DatabaseConnection, ObjectId
from groq_redis import GroqRedisService
from logger import ErrorLogger
from config import *

from redis_updater import ArticleRedisInsertion


warnings.filterwarnings("ignore")
logger = ErrorLogger(log_to_terminal=True)

app = FastAPI()
groqRedisService = GroqRedisService()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

MODEL = "gemma2-9b-it"
MODEL_local = "gemma2-9b-it_local"

def clean_article(text: str) -> str:
    # Function to clean the article text
    clean_text = re.sub(r'<[^<>]*>', '', text)
    clean_text = re.sub(r'http\S+| \n', ' ', clean_text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    clean_text = emoji_pattern.sub(r'', clean_text)
    clean_text_with_no_extra_spaces = [j for j in clean_text.split(" ") if j != ""]
    result = " ".join(clean_text_with_no_extra_spaces)
    return result

async def generate_summary(article: str, api_key: str):
    response = None
    try:
        prompt =  """ As you create the summary, follow these principles consistently to ensure accuracy and coherence:
                1. Detail and Complexity: Focus on crafting a summary that is in-depth and thorough, ensuring that all essential points from the text are covered comprehensively and also capture the numerical information from the article.
                2. Clarity and Conciseness: Maintain a consistent writing style that is clear and concise. Regularly check to eliminate any unnecessary words that don't contribute to the core message.
                3. Relevance: Consistently refer back to the provided text, ensuring that no external information is included in the summary. This keeps the summary aligned with the source material.
                4. Format: Consistently format the summary as a single, cohesive paragraph, making it easy to read and understand.
                5. Word Limit: Throughout the process, consistently check that the summary remains within the 60-word limit, ensuring it is both comprehensive and succinct.
                6. No Abrupt Ending: Summary should not have any abrupt ending; it must be completed, not leave the summary in between without completing it.
                7. Best Summary Selection: Generate multiple summaries for each article, and in the output, provide only the best summary that adheres to all the principles mentioned above.
                
                Only provide the summary without including phrases like "Summary:", special characters, or introductory statements like "Following the provided text, the best summary within the 60-word limit is:"
            """
            
        post_data = {
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content":  f"{article}"
                }
            ],
            "model": MODEL
        }
       
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        proxies = {
            'https': PROXY_SERVER.format(str(uuid4()).replace('-', ""))
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",  
                                headers=headers, 
                                data=json.dumps(post_data),
                                timeout=10,
                                proxies=proxies
                            )
        
        response.raise_for_status()
        res_headers = response.headers
        return response.json(), res_headers
    
    except Exception as e:
        logger.error(f"Error while summarizing - {e}")
        if response is not None and response.status_code in [400, 429, 503]:
            message = f"Error while summarizing article: \nAPI KEY:: {api_key} \nstatus Code:: {response.status_code} \n Message:: {response.text}"
            notify_slack(e, message)
        if(response):
            logger.error(f"Error while summarizing article: status Code:: {response.status_code} \n Message:: {response.text}")
        raise e
    

async def large_summarizer_groq(content):
    print("INFO:    Summarizing with chunking by `Langchain`")
    chunk_size = 2000
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=30)
    chunks = text_splitter.split_text(content)
    
    final_summary = ""
    model_used = ""
    for i, chunk in enumerate(chunks):
        print(f"INFO:    Summarizing chink {i+1}/{len(chunks)}")
        api_key = groqRedisService.get_best_api_key(chunk_size)
        response, header = generate_summary(chunk, api_key) 
        groqRedisService.set_rate_limit_info(api_key, header)
        summary = response['choices'][0]['message']['content']
        model_used = response['model']

        if summary:
            final_summary += summary + " "
    return final_summary, model_used


async def bart_summary(summary: str, bart_model, bart_tokenizer):
    # Function to re-summarize if needed
    model_name = 'facebook_bart_large_cnn'  # Model used for re-summarization

    start_time = time.time()
    print("INFO:    Resummarising with bart")
    if len(summary.split()) > 60:
        inputs = bart_tokenizer(summary, return_tensors='pt', max_length=1024, truncation=True).to(device)
        summary_ids = bart_model.generate(
            inputs['input_ids'],
            min_length=60,
            max_new_tokens=300,
            temperature=0.3,
            num_beams=6,
            length_penalty=2,
            no_repeat_ngram_size=2
        )
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    time_taken = time.time() - start_time
    return summary, model_name, time_taken


@app.head("/summarize_groq/")
def summarize_article_head(request: Request):
    return 'ok'


@app.post("/summarize_groq/")
async def summarize_article_groq(request: Request):
    articleRedisInsertion = ArticleRedisInsertion()
    articleRedisInsertion.setAioPikkaStatsSumm('received')
    try:
        start_time = time.time()
        max_summ_len = 60
        auth = request.headers.get("Authorization")
        
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        token = auth.split()[1]
        if token != BEARER_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Read the raw text from the request body
        input_text = await request.body()
        input_text = input_text.decode('utf-8')  # Decode bytes to string
        
        cleaned_article = clean_article(input_text)
        token_length = len(gpt2_tokenizer.encode(cleaned_article))
        
        summary = ""
        model_used = ""
        
        ## if(token_length > 6000):
        ##      raise HTTPException(status_code=400, detail="Token length is too long for groq.")
        if(token_length > 6000):
            summary, model_used = await large_summarizer_groq(cleaned_article)
            articleRedisInsertion.setAioPikkaStatsSumm('summary_langchain')
            articleRedisInsertion.setAioPikkaStatsSumm('summary_groq')
            summary, _, _ = await bart_summary(summary, bart_model, bart_tokenizer)
            articleRedisInsertion.setAioPikkaStatsSumm('summary_bart')
        else:
            api_key = groqRedisService.get_best_api_key(token_length)
            response, header = await generate_summary(input_text, api_key) 
            groqRedisService.set_rate_limit_info(api_key, header)
            summary = response['choices'][0]['message']['content']
            model_used = response['model']
            articleRedisInsertion.setAioPikkaStatsSumm('summary_groq')
        
        
        if summary is None or not summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary.")
        
        if(len(summary.split()) > max_summ_len):
            summary, _, _ = await bart_summary(summary, bart_model, bart_tokenizer)
            articleRedisInsertion.setAioPikkaStatsSumm('summary_bart')
            
        if summary is None or not summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary.")
        
        articleRedisInsertion.setAioPikkaStatsSumm('success')
        
        end_time = time.time()
        total_time_taken = end_time - start_time
        return {
            "summary": summary,
            "model_used": model_used,
            "time_taken": total_time_taken
        }
        
    except HTTPException as e:
        print(f"Error: {e}")
        articleRedisInsertion.setAioPikkaStatsSumm('summary_error')
        raise e

    except Exception as e:
        articleRedisInsertion.setAioPikkaStatsSumm('summary_error')
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
   

@app.post("/re-summarize-bf-bart/")
async def summarize_article(request: Request):
    try:
        auth = request.headers.get("Authorization")
        
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        token = auth.split()[1]
        if token != BEARER_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        
        # Read the raw text from the request body
        input_text = await request.body()
        input_text = input_text.decode('utf-8')  # Decode bytes to string
        
        if (not input_text):
            raise HTTPException(status_code=400, detail="Input text is empty") 
        
        summary, model_name, time_taken = await bart_summary(input_text, bart_model, bart_tokenizer)
        

        return {
            "summary": summary,
            "model_used": model_name,
            "time_taken": time_taken
        }
    except HTTPException as e:
        print(f"Error: {e}")
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")

   
   
    
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"status_code":404, "message": "The requested resource was not found on this server."}
    )
    
    

@app.on_event("shutdown")
def shutdown_event():
    notify_slack("Groq server is shutting down.", datetime.now().isoformat() , avoid_time_laps=True)

@app.on_event("startup")
async def list_routes_on_startup():
    print("Listing all registered routes:")
    for route in app.routes:
        if isinstance(route, APIRoute):
            methods = ", ".join(route.methods)
            print(f"INFO: {methods} {route.path}")


if __name__ == "__main__":
    try:
        uvicorn.run("__main__:app", host="0.0.0.0", port=6878, workers=3)
    except KeyboardInterrupt:
        print(f"Shutting down by user interaction")
    except Exception as e:
        error_traceback = traceback.format_exc()
        formatted_error = f"Error while summarizing articles:\n```\n{error_traceback}\n```"
        notify_slack(f"Groq summary server unable to start - {e}", formatted_error, avoid_time_laps=True)
        raise e