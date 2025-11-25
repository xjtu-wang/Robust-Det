import sys
import os
import pandas as pd
from tqdm import tqdm
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils_gen import get_prompt, count_tokens, truncate
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from openai import OpenAI
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
    client = OpenAI(base_url=url,api_key=api_key)
    return client

def call_openai_model(client, model_name, prompt, top_p, temp):
    """
    统一入口：根据模型类型调用 chat.completions 或 completions
    返回的是“续写的部分”（不包含原 prompt）
    """
    prompt_text = "Please continue this text in about 90 words: " + prompt.strip()

    if True:  # Use Chat Completions for all models in this script
        # Chat Completions
        messages = [{"role": "user", "content": prompt_text}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            top_p=top_p,
            temperature=temp,
            max_tokens=160,
        )
        ans = response.choices[0].message.content
    return ans

def get_prompt(text, prompt_len=20):
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens[:prompt_len])

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

    try_times = MAX_TRIAL
    outputs = []
    rep_tot, gen_tot = 0, 0
    for index, d in tqdm(df.iterrows()):
        if d["label"] == 1: # Human written texts
            continue
        seq = d["sequence"]
        prompt = get_prompt(seq)
                
        for idx in range(MAX_TRIAL):
            ans = call_openai_model(client, model_name, prompt, args.top_p, args.temp)

            # 你之前的逻辑：gpt-4 用 exp_truncate，其它用 truncate
            if model_name.startswith("gpt-4"):
                decoded_output = exp_truncate(prompt.strip() + " " + ans, 110)
            else:
                decoded_output = truncate(prompt.strip() + " " + ans, 110)

            token_num = count_tokens(decoded_output)
            print(f"*Attempt {idx}, #token {token_num}: {decoded_output[:80]}...")

            if token_num in range(70, 121):
                df.at[index, "sequence"] = decoded_output
                break

            if idx == MAX_TRIAL - 1:
                print("Token num not in range after max trials:", token_num)
                df.at[index, "sequence"] = "<blank text>"
        gen_tot += 1
        rep = evaluate_text(decoded_output)
        if rep:
            print(f"***Repeating {rep} times***")
            rep_tot += rep
            print(f"===Repeating tot {rep_tot/gen_tot}={rep_tot}/{gen_tot} times===")
        
        df.to_csv(out_path, sep = "|", index = None)
    print("Writing csv file to " + out_path)

if __name__ == "__main__":
    main()