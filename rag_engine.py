from transformers import pipeline

def generate_answer_with_flan_t5(query, contexts):
    prompt = f"Answer the question strictly based on the following context:\n\n{contexts}\n\nQuestion: {query}\nAnswer:"
    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    output = generator(prompt, max_length=250)
    return output[0]['generated_text']
