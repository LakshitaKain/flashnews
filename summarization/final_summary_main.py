import os
import time
import itertools
import re
import pandas as pd
from groq import Groq
from transformers import BartTokenizer, BartForConditionalGeneration, GPT2TokenizerFast
from typing import Optional, Tuple
from langchain.text_splitter import TokenTextSplitter
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import warnings
import json
import requests
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_fixed
# from Local_gemma import TextGenerator
import torch
from Local_gemma import TextGenerator
from logger import ErrorLogger
from groq_redis import GroqRedisService
from slack_notifier import notify_slack 
import traceback
from uuid import uuid4
from config import *


warnings.filterwarnings("ignore")
logger = ErrorLogger(log_to_terminal=True)

# Single model to use
MODEL = "gemma2-9b-it"
MODEL_local = "gemma2-9b-it_local"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

groqRedisService = GroqRedisService()

# Initialize BART model and tokenizer for re-summarization

# bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
# bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

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


def resummarize_if_needed(summary: str, bart_model, bart_tokenizer):
    # Function to re-summarize if needed
    model_name = 'facebook_bart_large_cnn'  # Model used for re-summarization

    start_time = time.time()

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



def generate_summary(article: str, api_key: str) -> Tuple[str, str, float]:
    response = None
    logger.info(f"Generating summary with groq API")
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
        # with open("post_data.json", "w") as f:
        #     x = json.dump(post_data, f)
        #     f.write(x)  
        #     f.close()
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
                                timeout=20,
                                proxies=proxies
                            )
        response.raise_for_status()
        res_headers = response.headers
        
        return response.json(), res_headers
    
    except Exception as e:
        logger.error(f"Error while summarizing article: status Code:: {response.status_code} \n Message:: {response.text}")
        if response is not None and response.status_code in [400, 429, 503]:
            message = f"Error while summarizing article: \nAPI KEY:: {api_key} \nstatus Code:: {response.status_code} \n Message:: {response.text}"
            notify_slack(e, message)
       
        raise e
    

def summarize_large_article(article: str, bart_model, bart_tokenizer):
    # Function to summarize large articles
    start_time = time.time()
    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=30)
    chunks = text_splitter.split_text(article)

    final_summary = ""
    for chunk in chunks:
        api_key = groqRedisService.get_best_api_key(2000)
        response, res_headers = generate_summary(chunk, api_key)
        groqRedisService.set_rate_limit_info(api_key, res_headers)
        summary = response['choices'][0]['message']['content']
        # summary, model_used, _ = generate_summary(chunk, api_key)
        summary, model_used, _ = resummarize_if_needed(summary, bart_model, bart_tokenizer)
        summary = clean_article(summary)
        if summary:
            final_summary += summary + " "
    
        # final_summary += f"Error summarizing chunk: {e}\n"

    final_summary = final_summary.strip()
    final_summary, model_used, _ = resummarize_if_needed(final_summary, bart_model, bart_tokenizer)
    time_taken = time.time() - start_time
    return final_summary, MODEL, time_taken


def summarize_local_gemma(article: str, bart_model, bart_tokenizer, gemma_model, gemma_tok):
    # Function to summarize using the local Gemma model
    start_time = time.time()
    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=30)
    chunks = text_splitter.split_text(article)

    final_summary = ""
    for chunk in chunks:
        gemma = TextGenerator()
        summary, model_used, _ = gemma.generate_text(chunk, gemma_model, gemma_tok)
        if summary:
            final_summary += summary + " "

    final_summary = final_summary.strip()
    final_summary, model_used, _ = resummarize_if_needed(final_summary, bart_model, bart_tokenizer)
    time_taken = time.time() - start_time
    return final_summary, MODEL_local, time_taken


def process_article_ex(article_content: str, bart_model, bart_tokenizer, gpt2_tokenizer, gemma_model, gemma_tok):
    # Main function to process an article
    start_time = time.time()

    if not article_content.strip():
        raise HTTPException(400, "Error 400 - Input is blank")


    try:
        
        cleaned_article = clean_article(article_content)
        token_length = len(gpt2_tokenizer.encode(cleaned_article))
        if token_length > 6000:
            return summarize_large_article(cleaned_article, bart_model, bart_tokenizer)
        else:
            api_key = groqRedisService.get_best_api_key(token_length)
            response, res_headers = generate_summary(cleaned_article, api_key)
            groqRedisService.set_rate_limit_info(api_key, res_headers)
            summary = response['choices'][0]['message']['content']
            model_used = response['model']
            # summary, model_used, time_taken = generate_summary_ex(cleaned_article, bart_model, bart_tokenizer)
            summary, _, _ = resummarize_if_needed(summary, bart_model, bart_tokenizer)
            summary = clean_article(summary)
            
            total_time_taken = time.time() - start_time
            return summary, model_used, total_time_taken

    except (HTTPError, Exception) as e:
        logger.error(f"{e}")
        logger.info(f"Now trying with local gemma model")
       
        # Use local gemma model if API fails
        try:
            
            cleaned_article = clean_article(article_content)
            token_length = len(gpt2_tokenizer.encode(cleaned_article))
            if token_length > 6000:
                return summarize_local_gemma(cleaned_article, bart_model, bart_tokenizer)
            else:
                start_time = time.time()
                gemma = TextGenerator()
                gemma_res = gemma.generate_text(cleaned_article, gemma_model, gemma_tok)
                sum_clean = clean_article(gemma_res)
                res, _, _ = resummarize_if_needed(sum_clean, bart_model, bart_tokenizer)

                time_taken = time.time() - start_time
                return res, MODEL_local, time_taken

        except Exception as gemma_e:
            raise HTTPException(500, f"Error 500 - Something went wrong: {gemma_e}")
        
        
def process_article(article_content: str, bart_model, bart_tokenizer, gpt2_tokenizer, gemma_model, gemma_tok):
    # Main function to process an article
    start_time = time.time()

    if not article_content.strip():
        raise HTTPException(400, "Error 400 - Input is blank")

    # Use local gemma model if API fails
    try:
        
        cleaned_article = clean_article(article_content)
        token_length = len(gpt2_tokenizer.encode(cleaned_article))
        if token_length > 6000:
            return summarize_local_gemma(cleaned_article, bart_model, bart_tokenizer, gemma_model, gemma_tok)
        else:
            start_time = time.time()
            gemma = TextGenerator()
            gemma_res = gemma.generate_text(cleaned_article, gemma_model, gemma_tok)
            sum_clean = clean_article(gemma_res)
            res, _, _ = resummarize_if_needed(sum_clean, bart_model, bart_tokenizer)

            time_taken = time.time() - start_time
            return res, MODEL_local, time_taken

    except Exception as gemma_e:
        raise HTTPException(500, f"Error 500 - Something went wrong: {gemma_e}")
        
        
        
        