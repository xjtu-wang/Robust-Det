import sys
import os
import pandas as pd
from tqdm import tqdm
from zai import ZhipuAiClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils_gen import get_prompt, count_tokens, truncate
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from openai import OpenAI,BadRequestError
import nltk
import random
from repeating_detect import evaluate_text 
CACHE_DIR = "./cache"
MAX_TRIAL = 10
SPLIT = "train"

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--domain",
        type=str,
        default="news",
        help="choose from 'news','review' and 'wiki'",
    ) 
    parser.add_argument(
        "--load_dataset_model",
        type=str,
        default="gptj",
        help="for loading dataset to get the human-written texts as reference, match with file name.",
    )
    parser.add_argument(
        "--gen_model_name",
        type=str,
        default="glm-4-flash",
        help="name for model or saved dataset, i.e., gpt-3.5-turbo-instruct, Llama2-7b-hf, gptj, text-davinci-003, gpt2-xl, ..."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="multi_model_data",
        help="csv file including positive samples",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="official model name in Huggingface/other API. Will generate automatically.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.96,
        help="",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="",
    )
    parser.add_argument(
        "--rp",
        type=float,
        default=1.0,
        help="",
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key, if using OpenAI models.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://open.bigmodel.cn/api/paas/v4/",
        help="Base URL for OpenAI API.",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=200,
        help="Maximum number of parallel requests.",
    )
    args = parser.parse_args()
    return args

    
def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

seed_everything(1)

#按句子截断到接近 tgt 个 token
def exp_truncate(text,tgt):
    now = 0
    sens = nltk.sent_tokenize(text)
    for i in range(len(sens)):
        if now < tgt:
            last = now
            now += count_tokens(sens[i])
        else:
            break
    if tgt-last < now-tgt:
        out_sens = i
    else:
        out_sens = i+1
    res = " ".join(sens[:out_sens])
    return res

def create_openai_client(args):
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("No OpenAI API key provided. Use --api_key or set OPENAI_API_KEY env var.")
    client = ZhipuAiClient(api_key=api_key)
    return client

def call_openai_model(client, model_name, prompt, top_p, temp):
    """
    统一入口：根据模型类型调用 chat.completions 或 completions
    返回的是“续写的部分”（不包含原 prompt）
    """
    prompt_text = "Please continue this text in about 180 words"+" and avoiding explicit descriptions of violence or hate speech: " + prompt.strip()

    if True:  # Use Chat Completions for all models in this script
        # Chat Completions
        messages = [        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ],
        }]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            top_p=top_p,
            temperature=temp,
            max_tokens=200,
        )
        ans = response.choices[0].message.content
    return ans

def get_prompt(text, prompt_len=20):
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens[:prompt_len])

def process_one_row(index, row, client, args, model_name):
    if row["label"] == 1:
        return index, row["sequence"], 0  # Human written texts, no need to generate

    seq = row["sequence"]

    # 如果原文太短或者是 NaN，直接跳过这条
    if not isinstance(seq, str) or len(seq.strip().split()) < 10:
        return index, "<blank text>", 0

    prompt = get_prompt(seq)

    decoded_output = "<blank text>"
    rep = 0

    for idx in range(MAX_TRIAL):
        try:
            ans = call_openai_model(client, model_name, prompt, args.top_p, args.temp)
        except BadRequestError as e:
            print(f"[idx {index}] Content blocked by API (BadRequestError): {e}")
            decoded_output = "<blocked by content filter>"
            rep = 0
            break
        except Exception as e:
            print(f"[idx {index}] Unexpected error: {e}")
            decoded_output = "<error>"
            rep = 0
            break        
        full_text = prompt.strip() + " " + ans
        if model_name.startswith("gpt-4") or "glm-4" in model_name:
            decoded_output = exp_truncate(full_text, 110)
        else:
            decoded_output = truncate(full_text, 110)

        token_num = count_tokens(decoded_output)
        print(f"[idx {index}] Attempt {idx}, #token {token_num}: {decoded_output[:80]}...")

        if token_num in range(35,121):   # 举例：随便写了个条件
            break

        if idx == MAX_TRIAL - 1:
            print(f"[idx {index}] Token num not in range after max trials: {token_num}")
            decoded_output = "<blank text>"

    # 重复检测
    rep = evaluate_text(decoded_output)
    if rep:
        print(f"[idx {index}] ***Repeating {rep} times***")

    return index, decoded_output, rep

def main(): 
    args = parse_args()
    client = create_openai_client(args)
    
    out_path = os.path.join(args.input_path, args.domain +"_"+ args.gen_model_name +"_t"+ str(args.temp)) + "/" + args.gen_model_name + f"_{SPLIT}.csv"
    if not os.path.exists(os.path.join(args.input_path, args.domain +"_"+ args.gen_model_name +"_t"+ str(args.temp))):
        os.mkdir(os.path.join(args.input_path, args.domain +"_"+ args.gen_model_name +"_t"+ str(args.temp)))
    print("*Output to file:", out_path)
    TESTSET_PATH = os.path.join(args.input_path, args.domain + "/" + args.load_dataset_model + f"_{SPLIT}.csv")

    df = pd.read_csv(TESTSET_PATH, sep="|")
    print("*Loaded from", TESTSET_PATH)

    model_name = args.gen_model_name
    MAX_LINES = args.max_lines

    try_times = MAX_TRIAL
    outputs = []
    rep_tot, gen_tot = 0, 0

    tasks = [(idx, row) for idx, row in df.iterrows() if row["label"] != 1]

    with ThreadPoolExecutor(max_workers=MAX_LINES) as executor:
        future_to_idx = {
            executor.submit(process_one_row, idx, row, client, args, model_name): idx
            for idx, row in tasks
        }

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            idx, decoded_output, rep = future.result()
            df.at[idx, "sequence"] = decoded_output

            if df.at[idx, "label"] != 1:
                gen_tot += 1
                if rep:
                    rep_tot += rep
                    print(f"===Repeating tot {rep_tot/gen_tot}={rep_tot}/{gen_tot} times===")


    df.to_csv(out_path, sep = "|", index = None)
    print("Writing csv file to " + out_path)

if __name__ == "__main__":
    main()