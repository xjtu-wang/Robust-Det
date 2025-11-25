import pandas as pd
from utils_gen import count_tokens

path = "multi_model_data/news_glm-4-flash_t1.0/glm-4-flash_test.csv"
df = pd.read_csv(path, sep="|")

mask = (df["label"] != 1) & (~df["sequence"].astype(str).str.startswith("<"))
sub = df[mask]

lengths = sub["sequence"].apply(count_tokens)

print("样本数：", len(sub))
print("平均 token 数：", lengths.mean())
print(lengths.describe())
