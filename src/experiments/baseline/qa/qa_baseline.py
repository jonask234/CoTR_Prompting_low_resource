# thesis_project/baseline/qa_baseline.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
import time
from collections import Counter

# Englische Beispiele für Few-Shot-Prompting
ENGLISH_FEW_SHOT_EXAMPLES_DIRECT_QA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"}
]

ENGLISH_FEW_SHOT_EXAMPLES_CONTEXTUAL_QA = [
    {
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "question": "Where is the Eiffel Tower located?",
        "answer": "Champ de Mars in Paris, France"
    },
    {
        "context": "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity.",
        "question": "What was Marie Curie's field of research?",
        "answer": "Radioactivity"
    },
    {
        "context": "The Amazon rainforest is a moist broadleaf tropical rainforest that covers most of the Amazon basin of South America.",
        "question": "How large is the Amazon rainforest?",
        "answer": "Approximately 5.5 million square kilometers"
    }
]

# Anweisungen für verschiedene Sprachen
LRL_QA_INSTRUCTIONS = {
    "sw": {
        "direct_instruction": "Tafadhali jibu swali lifuatalo.",
        "contextual_instruction": "Tafadhali jibu swali lifuatalo kulingana na muktadha uliotolewa.",
        "examples_header": "Hapa kuna mifano (mifano hii ni kwa Kiingereza):",
        "analyze_header": "Sasa, jibu swali hili:",
        "question_label": "Swali",
        "context_label": "Muktadha",
        "answer_label": "Jibu"
    },
    "fi": {
        "direct_instruction": "Vastaa seuraavaan kysymykseen.",
        "contextual_instruction": "Vastaa seuraavaan kysymykseen annetun kontekstin perusteella.",
        "examples_header": "Tässä esimerkkejä (nämä esimerkit ovat englanniksi):",
        "analyze_header": "Nyt vastaa tähän kysymykseen:",
        "question_label": "Kysymys",
        "context_label": "Konteksti",
        "answer_label": "Vastaus"
    }
}

# Allgemeine Anweisungen auf Englisch
GENERAL_INSTRUCTION_DIRECT_QA = "Please answer the following question."
GENERAL_INSTRUCTION_CONTEXTUAL_QA = "Please answer the following question based on the provided context."

def initialize_model(model_name):
    # Initialisiert ein Modell und einen Tokenizer
    print(f"Initializing {model_name}...")
    cache_path = "/work/bbd6522/cache_dir"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_path
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

def generate_qa_prompt(question, context=None, lang_code="en", use_few_shot=True, prompt_in_lrl=False):
    # Erzeugt einen Prompt für die QA-Aufgabe
    processed_question = question.strip()
    processed_context = context.strip() if context else None
    
    main_instruction = ""
    q_label, c_label, a_label = "Question", "Context", "Answer"
    examples_header_text = "Here are some examples (these examples are in English):"
    analyze_header_text = "Now, answer this question:"

    if prompt_in_lrl and lang_code in LRL_QA_INSTRUCTIONS:
        lrl_instr = LRL_QA_INSTRUCTIONS[lang_code]
        main_instruction = lrl_instr["contextual_instruction"] if processed_context else lrl_instr["direct_instruction"]
        q_label = lrl_instr["question_label"]
        c_label = lrl_instr["context_label"]
        a_label = lrl_instr["answer_label"]
        examples_header_text = lrl_instr["examples_header"]
        analyze_header_text = lrl_instr["analyze_header"]
    else:
        main_instruction = GENERAL_INSTRUCTION_CONTEXTUAL_QA if processed_context else GENERAL_INSTRUCTION_DIRECT_QA

    prompt = main_instruction
        
    if use_few_shot:
        prompt += f"\\n\\n{examples_header_text}\\n"
        examples_to_use = ENGLISH_FEW_SHOT_EXAMPLES_CONTEXTUAL_QA if processed_context else ENGLISH_FEW_SHOT_EXAMPLES_DIRECT_QA
        for ex in examples_to_use:
            prompt += f"\\n{q_label}: {ex['question']}\\n"
            if processed_context and 'context' in ex:
                prompt += f"{c_label}: {ex['context']}\\n"
            prompt += f"{a_label}: {ex['answer']}\\n"

    prompt += f"\\n\\n{analyze_header_text}\\n"
    prompt += f"{q_label}: {processed_question}\\n"
    if processed_context:
        prompt += f"{c_label}: {processed_context}\\n"
    prompt += f"{a_label}:"
    
    return prompt

def extract_answer(text):
    # Extrahiert die Antwort aus der Modellausgabe
    if not text or not text.strip():
        return "[no answer generated]"
    
    text = text.strip()
    
    # Entfernt gängige Phrasen, die keine Antworten sind
    prefixes_to_remove = ["answer:", "the answer is:", "jibu:", "vastaus:", "I don't know", "sijui", "en tiedä"]
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    # Entfernt Anführungszeichen
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    # Kürzt die Antwort auf den ersten Satz, wenn sie zu lang ist
    if len(text.split()) > 15:
        text = text.split('.')[0]
        
    return text.strip()

def process_qa_baseline(tokenizer, model, question, context=None, 
                          max_new_tokens=50,
                          temperature=0.3,
                          top_p=0.85, 
                          top_k=40,   
                          repetition_penalty=1.2, 
                          do_sample=True, 
                          lang_code="en",
                          use_few_shot=True,
                          prompt_in_lrl=False):
    # Verarbeitet eine einzelne QA-Anfrage
    start_time = time.time()

    prompt = generate_qa_prompt(
        question=question,
        context=context,
        lang_code=lang_code,
        use_few_shot=use_few_shot,
        prompt_in_lrl=prompt_in_lrl
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )
    
    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    answer = extract_answer(output_text)
    
    runtime = time.time() - start_time
    
    return answer, runtime, output_text

def calculate_qa_f1(ground_truth, predicted_answer):
    # Berechnet den F1-Score zwischen der korrekten und der vorhergesagten Antwort
    
    # Macht aus der korrekten Antwort eine Liste, falls sie es nicht ist
    if isinstance(ground_truth, str):
        references = [ground_truth]
    elif isinstance(ground_truth, dict) and 'text' in ground_truth:
        references = ground_truth['text']
    else:
        references = ground_truth
    
    prediction = str(predicted_answer)
    
    # Hilfsfunktion, um Text zu normalisieren (Kleinbuchstaben, Satzzeichen entfernen)
    def normalize_text(s):
        s = s.lower()
        s = re.sub(r'[\\p{P}\\p{S}]', '', s)
        s = re.sub(r'\\s+', ' ', s).strip()
        return s
    
    # Berechnet den F1-Score zwischen zwei Texten
    def f1_score(prediction_tokens, ground_truth_tokens):
        if not prediction_tokens or not ground_truth_tokens:
            return 0.0
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    # Vergleicht die Vorhersage mit jeder möglichen korrekten Antwort
    f1_scores = []
    for ref in references:
        if not ref:
            f1_scores.append(0.0)
            continue
        
        norm_ref = normalize_text(ref)
        norm_pred = normalize_text(prediction)
        
        f1 = f1_score(norm_pred.split(), norm_ref.split())
        f1_scores.append(f1)
    
    return max(f1_scores) if f1_scores else 0.0

def evaluate_qa_baseline(
    model_name,
    tokenizer,
    model,
    samples_df,
    lang_code,
    use_few_shot=True,
    prompt_in_lrl=False,
    temperature=0.3, 
    top_p=0.9,       
    top_k=40,        
    max_tokens=50,     
    repetition_penalty=1.2, 
    do_sample=True
):
    # Wertet die QA-Baseline für einen Datensatz aus
    results = []
    
    print(f"Starting QA baseline evaluation for {model_name} on {lang_code} "
          f"({'Few-shot' if use_few_shot else 'Zero-shot'}, "
          f"Prompt Instruction LRL: {prompt_in_lrl}).")

    for idx, row in samples_df.iterrows():
        question = str(row['question'])
        context = str(row.get('context', ''))
        ground_truth_answer = str(row.get('answers', '')) 

        predicted_answer, runtime_sample, raw_output = process_qa_baseline(
            tokenizer=tokenizer,
            model=model,
            question=question,
            context=context,
            lang_code=lang_code,
            use_few_shot=use_few_shot,
            prompt_in_lrl=prompt_in_lrl,
            temperature=temperature, 
            top_p=top_p,          
            top_k=top_k,
            max_new_tokens=max_tokens, 
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )
            
        f1_score_val = calculate_qa_f1(ground_truth_answer, predicted_answer)

        result = {
            "question": question,
            "ground_truth": ground_truth_answer,
            "predicted_answer": predicted_answer,
            "f1_score": f1_score_val,
            "language": lang_code,
            "shot_type": "few-shot" if use_few_shot else "zero-shot",
            "runtime_seconds": runtime_sample,
            "model": model_name,
            "raw_output": raw_output
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        avg_f1 = results_df["f1_score"].mean()
        avg_runtime = results_df["runtime_seconds"].mean()
        
        print(f"\\nResults for {lang_code} with {model_name} ({'few-shot' if use_few_shot else 'zero-shot'}):")
        print(f"  Average F1 score: {avg_f1:.4f}")
        print(f"  Average runtime: {avg_runtime:.2f} seconds")
    else:
        print(f"No results generated for {lang_code} with {model_name}.")
    
    return results_df