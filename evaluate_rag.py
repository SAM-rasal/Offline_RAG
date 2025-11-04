import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
import numpy as np

# Ensure required NLTK data is downloaded
nltk.download('punkt', quiet=True)

# ----------- STEP 1: Load & Clean Data -----------
FILE_NAME = "New_dataset.csv"
df = pd.read_csv(FILE_NAME)
df = df.rename(columns={'Ground_truth': 'ground_truth','Generated_answer': 'generated'})
df = df.dropna(subset=['ground_truth', 'generated'])
df['ground_truth'] = df['ground_truth'].astype(str).str.strip()
df['generated'] = df['generated'].astype(str).str.strip()
df = df[(df['ground_truth'].str.len() > 0) & (df['generated'].str.len() > 0)]
print(f"Evaluating {len(df)} valid samples.")

# ----------- STEP 2: Metrics Code -----------
rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth_func = SmoothingFunction().method1

results = []
for index, row in df.iterrows():
    ref = row['ground_truth']
    pred = row['generated']

    # BLEU
    bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_func)
    # ROUGE-L
    rouge_l = rouge_scorer_obj.score(ref, pred)['rougeL'].fmeasure
    # TF-IDF Cosine Similarity
    vect = TfidfVectorizer(stop_words='english').fit([ref, pred])
    tfidf = vect.transform([ref, pred])
    cos = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    # BERTScore (F1)
    P, R, F1 = bert_score([pred], [ref], lang="en", verbose=False, device='cpu')
    bert_f1 = F1.mean().item()

    results.append({
        'BLEU': bleu,
        'ROUGE-L': rouge_l,
        'Cosine Similarity': cos,
        'BERTScore': bert_f1
    })

# ----------- STEP 3: Print Overall Metrics -----------
metrics_df = pd.DataFrame(results)
avg_scores = metrics_df.mean(numeric_only=True)

print("\n===== RAG Model Aggregate Metrics =====")
for metric, score in avg_scores.items():
    print(f"{metric:<20}: {score:.4f}")
print("="*45)
