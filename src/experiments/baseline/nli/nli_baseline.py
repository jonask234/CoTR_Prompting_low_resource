from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import re
import os
from typing import Tuple, Dict, List, Any, Optional

# Define label mapping for NLI
NLI_LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

def initialize_model(model_name: str) -> Tuple:
    """
    Initialize a model for NLI task.
    
    Args:
        model_name: Name of the model to initialize
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    return model, tokenizer

def generate_nli_prompt(
    premise: str,
    hypothesis: str,
    lang_code: str = "en",
    use_few_shot: bool = True,
    prompt_in_lrl: bool = False
) -> str:
    """
    Generate a prompt for NLI task with improved structure.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        lang_code: Language code
        use_few_shot: Whether to include few-shot examples
        prompt_in_lrl: Whether to use instructions in the low-resource language
        
    Returns:
        A formatted NLI prompt
    """
    # Base instructions with clearer structure for all languages
    if lang_code == "en":
        instruction = """Text: '<PREMISE>'\nHypothesis: '<HYPOTHESIS>'\n\nDetermine if the hypothesis is ENTAILMENT, CONTRADICTION, or NEUTRAL with respect to the text. 

Definitions:
- ENTAILMENT: The hypothesis must be true if the text is true.
- CONTRADICTION: The hypothesis cannot be true if the text is true.
- NEUTRAL: The hypothesis might be true or false; the text doesn't provide enough information.

Provide your answer as EXACTLY one of these three English labels: 'ENTAILMENT', 'CONTRADICTION', or 'NEUTRAL'.
Your entire response must be only one of these three words. No other text, explanation, or punctuation is allowed."""
    elif lang_code == "sw":  # Swahili
        instruction = """Maandishi: '<PREMISE>'\nHipothesia: '<HYPOTHESIS>'\n\nAmua kama hipothesia ni ENTAILMENT, CONTRADICTION, au NEUTRAL kuhusiana na maandishi.

Ufafanuzi:
- ENTAILMENT: Hipothesia lazima iwe kweli ikiwa maandishi ni kweli.
- CONTRADICTION: Hipothesia haiwezi kuwa kweli ikiwa maandishi ni kweli.
- NEUTRAL: Hipothesia inaweza kuwa kweli au si kweli kutokana na dhana

Jibu kwa kutumia MOJA TU ya lebo hizi za Kiingereza: 'ENTAILMENT', 'CONTRADICTION', au 'NEUTRAL'.
Jibu lako lote lazima liwe neno MOJA TU kati ya hayo matatu ya Kiingereza. Hakuna maandishi mengine, maelezo, au alama za uakifishaji zinazoruhusiwa."""
    elif lang_code == "ur":  # Urdu
        instruction = """متن: '<PREMISE>'\nمفروضہ: '<HYPOTHESIS>'\n\nیہ تعین کریں کہ مفروضہ متن کے حوالے سے ENTAILMENT، CONTRADICTION، یا NEUTRAL ہے۔

تعریفیں:
- ENTAILMENT: اگر متن سچ ہے تو مفروضہ بھی ضرور سچ ہے۔
- CONTRADICTION: اگر متن سچ ہے تو مفروضہ سچ نہیں ہو سکتا۔
- NEUTRAL: مفروضہ سچ یا غلط ہو سکتا ہے؛ متن کافی معلومات فراہم نہیں کرتا۔

اپنا جواب صرف ان تین انگریزی لیبلز میں سے ایک کے طور پر دیں: 'ENTAILMENT'، 'CONTRADICTION'، یا 'NEUTRAL'۔
آپ کا پورا جواب صرف ان تین الفاظ میں سے ایک ہونا چاہیے۔ کسی دوسرے متن، وضاحت یا رموز اوقاف کی اجازت نہیں ہے۔"""
    else:  # Default to English
        instruction = """Text: '<PREMISE>'\nHypothesis: '<HYPOTHESIS>'\n\nDetermine if the hypothesis is ENTAILMENT, CONTRADICTION, or NEUTRAL with respect to the text. 

Definitions:
- ENTAILMENT: The hypothesis must be true if the text is true.
- CONTRADICTION: The hypothesis cannot be true if the text is true.
- NEUTRAL: The hypothesis might be true or false; the text doesn't provide enough information.

Provide your answer as EXACTLY one of these three English labels: 'ENTAILMENT', 'CONTRADICTION', or 'NEUTRAL'.
Your entire response must be only one of these three words. No other text, explanation, or punctuation is allowed."""

    # Few-shot examples (language-specific if available)
    few_shot_examples = ""
    
    if use_few_shot:
        # English examples
        if lang_code == "en":
            few_shot_examples = """
Example 1:
Text: 'The chef is cooking a meal in the kitchen.'
Hypothesis: 'The chef is preparing food.'
Answer: ENTAILMENT

Example 2:
Text: 'The boy is playing soccer in the park.'
Hypothesis: 'The boy is swimming in a pool.'
Answer: CONTRADICTION

Example 3:
Text: 'The woman is walking down the street.'
Hypothesis: 'She is going to the grocery store.'
Answer: NEUTRAL

"""
        # Swahili examples
        elif lang_code == "sw":
            few_shot_examples = """
Mfano 1:
Maandishi: 'Mpishi anaandaa chakula jikoni.'
Hipothesia: 'Mpishi anatayarisha chakula.'
Jibu: ENTAILMENT

Mfano 2:
Maandishi: 'Mvulana anacheza mpira wa miguu katika bustani.'
Hipothesia: 'Mvulana anaogelea katika bwawa.'
Jibu: CONTRADICTION

Mfano 3:
Maandishi: 'Mwanamke anatembea barabarani.'
Hipothesia: 'Anaelekea dukani.'
Jibu: NEUTRAL

"""
        # Urdu examples
        elif lang_code == "ur":
            few_shot_examples = """
مثال 1:
متن: 'باورچی باورچی خانے میں کھانا پکا رہا ہے۔'
مفروضہ: 'باورچی کھانا تیار کر رہا ہے۔'
جواب: ENTAILMENT

مثال 2:
متن: 'لڑکا پارک میں فٹبال کھیل رہا ہے۔'
مفروضہ: 'لڑکا سوئمنگ پول میں تیر رہا ہے۔'
جواب: CONTRADICTION

مثال 3:
متن: 'ایک عورت گلی میں چل رہی ہے۔'
مفروضہ: 'وہ گروسری سٹور جا رہی ہے۔'
جواب: NEUTRAL

"""
        # Default to English examples if language-specific ones not available
        else:
            few_shot_examples = """
Example 1:
Text: 'The chef is cooking a meal in the kitchen.'
Hypothesis: 'The chef is preparing food.'
Answer: ENTAILMENT

Example 2:
Text: 'The boy is playing soccer in the park.'
Hypothesis: 'The boy is swimming in a pool.'
Answer: CONTRADICTION

Example 3:
Text: 'The woman is walking down the street.'
Hypothesis: 'She is going to the grocery store.'
Answer: NEUTRAL

"""
    
    # Replace instruction's placeholder with actual premise/hypothesis
    instruction = instruction.replace('<PREMISE>', premise).replace('<HYPOTHESIS>', hypothesis)
    
    # Combine into the final prompt
    if use_few_shot:
        final_prompt = f"{instruction}\n\n{few_shot_examples}\nAnswer:"
    else:
        final_prompt = f"{instruction}\n\nAnswer:"
    
    return final_prompt

def generate_lrl_instruct_nli_prompt(premise: str, hypothesis: str, lang_code: str = "sw", model_name: str = "", use_few_shot: bool = True) -> str:
    """
    Generate a prompt for NLI with instructions in the low-resource language.
    
    Args:
        premise: The premise text in LRL
        hypothesis: The hypothesis text in LRL
        lang_code: Language code (e.g., "sw" for Swahili)
        model_name: Name of the model for model-specific adjustments
        use_few_shot: Whether to include few-shot examples in the prompt
        
    Returns:
        A formatted prompt with instructions in the low-resource language
    """
    # Base instructions in different languages
    if lang_code == "sw":  # Swahili
        instruction = """Amua ikiwa dhana inathihirisha wazo, inapingana nalo, au si vyovyote (katikati).
Jibu kwa kutumia MOJA TU ya lebo hizi za Kiingereza: 'entailment', 'contradiction', au 'neutral'.
- 'entailment' inamaanisha kuwa wazo ni kweli kabisa kutokana na dhana
- 'contradiction' inamaanisha kuwa wazo si kweli kabisa kutokana na dhana
- 'neutral' inamaanisha kuwa wazo linaweza kuwa kweli au si kweli kutokana na dhana
Jibu lako lazima liwe neno MOJA TU kati ya hayo matatu ya Kiingereza. Hakuna maandishi mengine, maelezo, au alama za uakifishaji zinazoruhusiwa."""
    elif lang_code == "ha":  # Hausa
        instruction = """Ƙyale ko magana tana tabbatar da ra'ayi, ko tana sabawa da shi, ko ba kowa (neutral).
Amsa da DAYA KAWAI daga cikin waɗannan lakabin Turanci: 'entailment', 'contradiction', ko 'neutral'.
- 'entailment' na nufin ra'ayi tabbas gaskiya ne bisa ga magana
- 'contradiction' na nufin ra'ayi tabbas ƙarya ne bisa ga magana
- 'neutral' na nufin ra'ayi zai iya zama gaskiya ko ƙarya bisa ga magana
Amsarka dole ta zama DAYA KAWAI daga cikin waɗannan kalmomi uku na Turanci. Ba a yarda da wani rubutu, bayani, ko alamar rubutu ba."""
    elif lang_code == "te":  # Telugu
        instruction = """ప్రతిపాదన పరికల్పనను అంగీకరిస్తోందో, వ్యతిరేకిస్తోందో లేదా ఏదీ కాదు (neutral) అని నిర్ణయించండి.
ఈ మూడు ఇంగ్లీష్ లేబుల్‌లలో ఖచ్చితంగా ఒకదానితో మాత్రమే సమాధానం ఇవ్వండి: 'entailment', 'contradiction', లేదా 'neutral'.
- 'entailment' అంటే ప్రతిపాదన ప్రకారం పరికల్పన ఖచ్చితంగా సత్యం
- 'contradiction' అంటే ప్రతిపాదన ప్రకారం పరికల్పన ఖచ్చితంగా అసత్యం
- 'neutral' అంటే ప్రతిపాదన ప్రకారం పరికల్పన సత్యం కావచ్చు లేదా కాకపోవచ్చు
మీ సమాధానం ఈ మూడు ఆంగ్ల పదాలలో ఖచ్చితంగా ఒకటి మాత్రమే అయి ఉండాలి. ఇతర వచనం, వివరణ లేదా విరామచిహ్నాలకు అనుమతి లేదు."""
    else:  # Default to English
        instruction = """Determine whether the premise entails the hypothesis, contradicts it, or neither (neutral).
Answer with EXACTLY one of the following English labels: 'entailment', 'contradiction', or 'neutral'.
- 'entailment' means the hypothesis is definitely true given the premise
- 'contradiction' means the hypothesis is definitely false given the premise
- 'neutral' means the hypothesis may or may not be true given the premise
Your entire response must be only one of these three English words. No other text, explanation, or punctuation is allowed."""

    # Examples in the appropriate language - only included in few-shot mode
    examples_text = ""
    
    if use_few_shot:
        # Indent this entire block defining examples
    if lang_code == "sw":  # Swahili
            examples_text = """Mifano:
Dhana: 'Mwanamke alivaa koti jekundu amesimama katika kituo cha basi.'
Wazo: 'Mwanamke alivaa koti jekundu.'
Jibu: entailment

Dhana: 'Msichana anacheza na mbwa wake kwenye uwanja wa majani.'
Wazo: 'Msichana yuko pwani akiogelea baharini.'
Jibu: contradiction

Dhana: 'Wanafunzi wanasoma vitabu vyao darasani.'
Wazo: 'Wanafunzi wanapenda kusoma.'
Jibu: neutral

Dhana: 'Mkulima anawanywesha maji mazao yake.'
Wazo: 'Mazao yanapata maji kutoka kwa mkulima.'
Jibu: entailment
"""
    elif lang_code == "ha":  # Hausa
            examples_text = """Misalai:
Magana: 'Wata mata mai sa jar riga tana tsaye a tsutsar bas.'
Ra'ayi: 'Wata mata ta sa jar riga.'
Amsa: entailment

Magana: 'Yarinya tana wasa da kare a filin ciyawa.'
Ra'ayi: 'Yarinya tana iyo a bakin teku.'
Amsa: contradiction

Magana: 'Ɗalibai suna karatu a aji.'
Ra'ayi: 'Ɗalibai suna jin daɗin karatu.'
Amsa: neutral

Magana: 'Wani manomi yana shayar da gonakinsa ruwa.'
Ra'ayi: 'Gonaki suna samun ruwa daga manomi.'
Amsa: entailment
"""
    elif lang_code == "te":  # Telugu
            examples_text = """ఉదాహరణలు:
ప్రతిపాదన: 'ఒక మహిళ ఎరుపు కోటు వేసుకుని బస్ స్టాండులో నిలబడి ఉంది.'
పరికల్పన: 'ఒక మహిళ ఎరుపు కోటు వేసుకుంది.'
సమాధానం: entailment

ప్రతిపాదన: 'ఒక అమ్మాయి పచ్చిక బయలులో తన కుక్కతో ఆడుతోంది.'
పరికల్పన: 'ఒక అమ్మాయి సముద్రంలో ఈదుతోంది.'
సమాధానం: contradiction

ప్రతిపాదన: 'విద్యార్థులు తరగతిలో తమ పుస్తకాలు చదువుతున్నారు.'
పరికల్పన: 'విద్యార్థులకు చదవడం అంటే ఇష్టం.'
సమాధానం: neutral

ప్రతిపాదన: 'ఒక రైతు తన పంటలకు నీరు పోస్తున్నాడు.'
పరికల్పన: 'పంటలు రైతు నుండి నీరు పొందుతున్నాయి.'
సమాధానం: entailment
"""
    else:  # Default English examples
            examples_text = """Examples:
Premise: 'A woman wearing a red coat is standing at a bus stop.'
Hypothesis: 'A woman is wearing a red coat.'
Answer: entailment

Premise: 'A girl is playing with her dog in a grassy field.'
Hypothesis: 'A girl is at the beach swimming in the ocean.'
Answer: contradiction

Premise: 'Students are reading their books in class.'
Hypothesis: 'The students enjoy reading.'
Answer: neutral

Premise: 'A farmer is watering his crops in the field.'
Hypothesis: 'The crops are receiving water from the farmer.'
Answer: entailment
"""

    # Final prompt in the appropriate language format
    if lang_code == "sw":  # Swahili format
        prompt = f"Dhana: '{premise}'\nWazo: '{hypothesis}'\n\n{instruction}\n\n{examples_text}\nJibu:"
    elif lang_code == "ha":  # Hausa format
        prompt = f"Magana: '{premise}'\nRa'ayi: '{hypothesis}'\n\n{instruction}\n\n{examples_text}\nAmsa:"
    elif lang_code == "te":  # Telugu format
        prompt = f"ప్రతిపాదన: '{premise}'\nపరికల్పన: '{hypothesis}'\n\n{instruction}\n\n{examples_text}\nసమాధానం:"
    else:  # Default English format
        prompt = f"Premise: '{premise}'\nHypothesis: '{hypothesis}'\n\n{instruction}\n\n{examples_text}\nAnswer:"
    
    return prompt

def extract_nli_label(output_text, normalize=True):
    """
    Extract NLI label from model output text with improved accuracy.
    
    Args:
        output_text: Raw text generated by the model
        normalize: Whether to normalize the label
        
    Returns:
        Extracted label: one of "entailment", "contradiction", "neutral"
    """
    # Convert to lowercase for matching and strip whitespace
    text = output_text.lower().strip()
    
    # First, look for capitalized labels that might be direct responses
    # from the structured prompt with exactly matched labels
    if "ENTAILMENT" in output_text:
        return "entailment"
    elif "CONTRADICTION" in output_text:
        return "contradiction"
    elif "NEUTRAL" in output_text:
        return "neutral"
    
    # If no exact capitalized match, look for patterns in the output
    # Try to find the label closest to the start of the response (highest priority)
    label_positions = {
        'entailment': text.find('entailment'),
        'contradiction': text.find('contradiction'),
        'neutral': text.find('neutral')
    }
    
    # Filter out labels that aren't present (-1)
    label_positions = {k: v for k, v in label_positions.items() if v != -1}
    
    if label_positions:
        # Return the label that appears first in the text
        return min(label_positions.items(), key=lambda x: x[1])[0]
    
    # Look for abbreviated or partial matches
    if 'enta' in text or 'entail' in text:
        return 'entailment'
    elif 'contra' in text or 'contrad' in text:
        return 'contradiction'
    elif 'neut' in text:
        return 'neutral'
    
    # Look for common synonyms or alternative expressions
    if any(term in text for term in ['yes', 'follows', 'must be true', 'is true', 'has to be true']):
        return 'entailment'
    elif any(term in text for term in ['no', 'not true', 'cannot be true', 'opposite', 'disagree']):
        return 'contradiction'
    elif any(term in text for term in ['maybe', 'unknown', 'not enough', 'insufficient', 'could be']):
        return 'neutral'
    
    # Additional language-specific matches
    # Swahili
    if 'lazima' in text or 'ni kweli' in text:  # Must be true, is true
        return 'entailment'
    elif 'haiwezi' in text or 'si kweli' in text:  # Cannot be, is not true
        return 'contradiction'
    elif 'inaweza' in text or 'huenda' in text:  # Might be, perhaps
        return 'neutral'
    
    # Hausa
    if 'dole' in text or 'gaskiya' in text:  # Must, truth
        return 'entailment'
    elif 'ba zai' in text or 'ƙarya' in text:  # Cannot, false
        return 'contradiction'
    elif 'iya zama' in text or 'wata' in text:  # Can be, some
        return 'neutral'
    
    # Default to most common label or neutral if nothing found
    return 'contradiction'

def process_nli_baseline(
    model,
    tokenizer, 
    premise, 
    hypothesis, 
    lang_code="en",
    temperature=0.7,
    max_new_tokens=30,
    do_sample=True,
    use_few_shot=True,
    prompt_in_lrl=False,
    top_p: float = 0.92,
    top_k: int = 40,
    repetition_penalty: float = 1.0
):
    """Process a single NLI example using the improved baseline approach."""
    import time
    import torch
    
    # Generate the prompt
    start_time = time.time()
    
    prompt = generate_nli_prompt(
        premise, 
        hypothesis, 
        lang_code=lang_code, 
        use_few_shot=use_few_shot,
        prompt_in_lrl=prompt_in_lrl
    )
    
    # Debug: Print the prompt being used (truncated for readability)
    print(f"\nPrompt for NLI ({lang_code}):")
    print(f"{prompt[:200]}... (truncated)")
        
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Adjust parameters by language for better results
    current_repetition_penalty = repetition_penalty # Use passed-in value
    current_temperature = temperature
    current_top_p = top_p
    current_top_k = top_k

    if lang_code != "en":
        # START Indented block for the 'if'
        current_temperature = max(0.08, temperature * 0.8)
        current_top_p = min(0.95, top_p * 1.05)  # Slightly higher top_p for more diversity
        current_repetition_penalty = repetition_penalty * 1.1  # Slightly higher to prevent repetition
        # END Indented block for the 'if'
    
    # This 'with' block should be at the same indentation level as the 'if' statement above.
        with torch.no_grad():
        outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            temperature=current_temperature,
            do_sample=do_sample,
            top_p=current_top_p,
            top_k=current_top_k,
            repetition_penalty=current_repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
            )
        
    # Get the generated text
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = response.strip()
    
    # Debug: Print raw model response
    print(f"\nRaw model response: '{response}'")
    
    # Extract the predicted label
    predicted_label = extract_nli_label(response)
    
    # Debug: Print extracted label
    print(f"Extracted label: '{predicted_label}'")
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    return {
        "predicted_label": predicted_label,
        "raw_response": response,
        "runtime_seconds": runtime
    }

def evaluate_nli_baseline(
    model_name,
    samples_df,
    lang_code,
    prompt_in_lrl=False,
    temperature=0.3,
    max_new_tokens=20,
    do_sample=False,
    use_few_shot=True,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.0
):
    """
    Evaluate NLI performance using the baseline approach.
    
    Args:
        model_name: Name of the model to use
        samples_df: DataFrame containing NLI samples
        lang_code: Language code
        prompt_in_lrl: Whether to use low-resource language for prompt instructions
        temperature: Temperature for generation
        max_new_tokens: Maximum new tokens to generate
        do_sample: Whether to use sampling instead of greedy decoding
        use_few_shot: Whether to use few-shot examples
        top_p: Top-p parameter for generation
        top_k: Top-k parameter for generation
        repetition_penalty: Repetition penalty for generation
        
    Returns:
        DataFrame with results
    """
    import torch
    import time
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nInitializing {model_name}...")
    
    # Initialize model
    model, tokenizer = initialize_model(model_name)
    model.eval()  # Set to evaluation mode
    
    # Process all samples
    results = []
    start_time = time.time()
    
    # Print sample counts
    print(f"Processing {len(samples_df)} samples for {lang_code}...")
    
    # Check if we have original_lang information in the dataset
    has_original_lang = 'original_lang' in samples_df.columns
    if has_original_lang:
        print(f"Found 'original_lang' column in dataset - will use actual language for text if different from {lang_code}")
        # Check if there are any fallbacks
        fallback_count = samples_df['is_fallback'].sum() if 'is_fallback' in samples_df.columns else 0
        if fallback_count > 0:
            print(f"WARNING: {fallback_count} samples are using fallback language rather than {lang_code}!")
    
    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=f"Processing {lang_code} samples"):
        premise = row['premise']
        hypothesis = row['hypothesis']
        ground_truth = row['label']  # Use consistent column name
        
        # Use original_lang for processing if available, otherwise use lang_code
        # This ensures we're using the appropriate language instructions for the actual text
        text_lang = row['original_lang'] if has_original_lang else lang_code
        
        # Process the NLI example
        result = process_nli_baseline(
            model,
                tokenizer, 
                premise, 
                hypothesis, 
            lang_code=text_lang,  # Use the actual language of the text
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_few_shot=use_few_shot,
            prompt_in_lrl=prompt_in_lrl,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
            )
            
        # Add sample and result information
        result.update({
                "premise": premise,
                "hypothesis": hypothesis,
            "gold_label": ground_truth,  # Use consistent column name for metrics calculation
            "language": lang_code,      # The requested language
            "text_language": text_lang, # The actual language of the text
            "prompt_language": "LRL" if prompt_in_lrl else "EN",
            "is_fallback": row.get('is_fallback', False) if has_original_lang else False
        })
        
            results.append(result)
    
    # Create results DataFrame
    total_runtime = time.time() - start_time
    results_df = pd.DataFrame(results)
    
    # Debug: Check the columns in the DataFrame
    print("\nColumns in results DataFrame:")
    for col in results_df.columns:
        print(f"  - {col}")
    
    # Debug: Check a sample of predictions
    print("\nSample of predictions:")
    if not results_df.empty:
        sample_size = min(5, len(results_df))
        for i in range(sample_size):
            row = results_df.iloc[i]
            print(f"\nExample {i+1}:")
            print(f"  Premise: {row['premise'][:50]}...")
            print(f"  Hypothesis: {row['hypothesis'][:50]}...")
            print(f"  Gold label: {row['gold_label']}")
            print(f"  Predicted label: {row['predicted_label']}")
            if has_original_lang:
                print(f"  Text language: {row['text_language']} (requested: {row['language']})")
            print(f"  Raw response: {row['raw_response']}")
    
    # Add experiment information
    results_df['model'] = model_name
    results_df['temperature'] = temperature
    results_df['max_new_tokens'] = max_new_tokens
    results_df['do_sample'] = do_sample
    results_df['top_p'] = top_p
    results_df['top_k'] = top_k
    results_df['repetition_penalty'] = repetition_penalty
    results_df['runtime_seconds'] = total_runtime
    results_df['runtime_per_sample'] = total_runtime / len(samples_df) if len(samples_df) > 0 else 0
    results_df['few_shot'] = use_few_shot
    
    return results_df

def calculate_nli_metrics(results_df):
    """
    Calculate accuracy and F1 scores for NLI predictions.
    
    Args:
        results_df: DataFrame with 'predicted_label' and 'gold_label' columns
    
    Returns:
        Dictionary with accuracy and F1 metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    # Debug: Print the first few rows to check format
    print("\nFirst 5 rows of results DataFrame:")
    print(results_df[['premise', 'hypothesis', 'gold_label', 'predicted_label']].head().to_string())
    
    # Count label distribution for both gold and predicted
    print("\nGold label distribution:")
    print(results_df['gold_label'].value_counts())
    
    print("\nPredicted label distribution:")
    print(results_df['predicted_label'].value_counts())
    
    # Make sure we have valid labels to calculate metrics
    if 'predicted_label' not in results_df.columns or 'gold_label' not in results_df.columns:
        print("ERROR: Missing 'predicted_label' or 'gold_label' columns in results DataFrame")
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'class_metrics': {}}
    
    if results_df['predicted_label'].isnull().all() or results_df['gold_label'].isnull().all():
        print("ERROR: All null values in 'predicted_label' or 'gold_label' columns")
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'class_metrics': {}}
    
    # Create a copy to avoid modifying the original DataFrame
    eval_df = results_df.copy()
    
    # Check if gold_label is numeric and predicted_label is string
    if pd.api.types.is_numeric_dtype(eval_df['gold_label']) and pd.api.types.is_string_dtype(eval_df['predicted_label']):
        print("Converting numeric gold labels to strings for compatibility...")
        # Map from numeric labels to string labels
        label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }
        # Convert gold_label from numeric to string format
        eval_df['gold_label'] = eval_df['gold_label'].map(label_map)
        print("After conversion, gold label distribution:")
        print(eval_df['gold_label'].value_counts())
    
    # Calculate accuracy
    accuracy = accuracy_score(eval_df['gold_label'], eval_df['predicted_label'])
    
    # Calculate macro F1 score
    macro_f1 = f1_score(
        eval_df['gold_label'], 
        eval_df['predicted_label'], 
        average='macro', 
        zero_division=0
    )
    
    # Get detailed metrics for each class
    class_report = classification_report(
        eval_df['gold_label'], 
        eval_df['predicted_label'], 
        output_dict=True, 
        zero_division=0
    )
    
    # Extract class metrics
    class_metrics = {}
    for label in ["entailment", "neutral", "contradiction"]:
        if label in class_report:
            class_metrics[label] = {
                'precision': class_report[label]['precision'],
                'recall': class_report[label]['recall'],
                'f1': class_report[label]['f1-score']
        }
    
    # Print classification report for debugging
    print("\nDetailed Classification Report:")
    print(classification_report(eval_df['gold_label'], eval_df['predicted_label'], zero_division=0))
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics
    } 