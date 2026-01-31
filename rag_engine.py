from transformers import pipeline

def generate_answer_with_flan_t5(query, contexts):
    prompt = f"You are a medical research assistant providing evidence-based answers. "
    "Your responses must be grounded STRICTLY in the provided context.\n\n"
    
    "CORE PRINCIPLES:\n"
    "• Factual accuracy: Use only information explicitly stated in the context\n"
    "• Completeness: Provide comprehensive answers with all relevant details\n"
    "• Precision: Include specific numbers, percentages, metrics, and technical terms\n"
    "• Scientific rigor: Maintain exact terminology and methodological accuracy\n\n"
    
    "ANSWERING RULES:\n" \
         "1. Use only facts explicitly stated in the context.\n" \
         "2. Do NOT guess, assume, or add new information.\n" \
         "3. Preserve all scientific and technical terms exactly.\n" \
         "4. Include all relevant numbers, percentages, sample sizes, and statistical values.\n" \
         "5. Write the answer in 2–3 complete, well-structured sentences.\n" \
         "6. Maintain formal scientific language.\n" \
         "7. If the context does not contain the required information, reply exactly:\n" \
         "\"Insufficient information in context.\"\n"
    
    f"DOCUMENT CONTEXT:\n{contexts}\n\n"
    f"QUESTION: {query}\n\n"
    "EVIDENCE-GROUNDED ANSWER:"
    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    output = generator(prompt, max_length=250)
    return output[0]['generated_text']
