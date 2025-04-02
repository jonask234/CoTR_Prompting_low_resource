# src/pipelines/cotr_pipeline.py

class CoTRPipeline:
    def __init__(self, model, tokenizer, translator, src_lang, tgt_lang="en"):
        """
        Initialize the Chain of Translation (CoTR) pipeline.
        
        Args:
            model: The language model to use.
            tokenizer: The tokenizer for the language model.
            translator: The translator to use (MarianMT or Google Translate).
            src_lang (str): Source language code.
            tgt_lang (str, optional): Target language code. Defaults to "en".
        """
        self.model = model
        self.tokenizer = tokenizer
        self.translator = translator
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def process(self, inputs, task, language):
        """
        Process inputs using the Chain of Translation approach.
        
        Args:
            inputs (dict): Input data for the task.
            task (str): The NLP task to perform.
            language (str): The language of the inputs.
            
        Returns:
            dict: Model outputs and translation information.
        """
        # Step 1: Translate inputs to English
        translated_inputs = self._translate_inputs(inputs, task, self.src_lang, self.tgt_lang)
        
        # Step 2: Process in English
        english_prompts = self._construct_prompts(translated_inputs, task, self.tgt_lang)
        
        english_outputs = []
        for prompt in english_prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(
                input_ids,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            english_outputs.append(output)
        
        # Step 3: Translate outputs back to source language
        translated_outputs = self._translate_outputs(english_outputs, task, self.tgt_lang, self.src_lang)
        
        return {
            "task": task,
            "language": language,
            "inputs": inputs,
            "translated_inputs": translated_inputs,
            "english_outputs": english_outputs,
            "outputs": translated_outputs
        }
    
    def _translate_inputs(self, inputs, task, src_lang, tgt_lang):
        """
        Translate inputs from source language to target language.
        
        Args:
            inputs (dict): Input data for the task.
            task (str): The NLP task to perform.
            src_lang (str): Source language code.
            tgt_lang (str): Target language code.
            
        Returns:
            dict: Translated inputs.
        """
        translated_inputs = []
        
        if task == "QA":
            for item in inputs:
                translated_context = self.translator.translate(item['context'], src_lang, tgt_lang)
                translated_question = self.translator.translate(item['question'], src_lang, tgt_lang)
                translated_inputs.append({
                    'context': translated_context,
                    'question': translated_question
                })
        
        elif task in ["Summarization", "NER", "Sentiment Analysis"]:
            for item in inputs:
                translated_text = self.translator.translate(item['text'], src_lang, tgt_lang)
                translated_inputs.append({
                    'text': translated_text
                })
        
        return translated_inputs
    
    def _translate_outputs(self, outputs, task, src_lang, tgt_lang):
        """
        Translate outputs from source language to target language.
        
        Args:
            outputs (list): Model outputs.
            task (str): The NLP task to perform.
            src_lang (str): Source language code.
            tgt_lang (str): Target language code.
            
        Returns:
            list: Translated outputs.
        """
        translated_outputs = []
        
        for output in outputs:
            translated_output = self.translator.translate(output, src_lang, tgt_lang)
            translated_outputs.append(translated_output)
        
        return translated_outputs
    
    def _construct_prompts(self, inputs, task, language):
        """
        Construct prompts for the model based on the task and language.
        
        Args:
            inputs (dict): Input data for the task.
            task (str): The NLP task to perform.
            language (str): The language of the inputs.
            
        Returns:
            list: List of prompts.
        """
        prompts = []
        
        if task == "QA":
            for item in inputs:
                prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer: "
                prompts.append(prompt)
        
        elif task == "Summarization":
            for item in inputs:
                prompt = f"Summarize the following text:\n{item['text']}\nSummary: "
                prompts.append(prompt)
        
        elif task == "NER":
            for item in inputs:
                prompt = f"Identify the named entities in the following text:\n{item['text']}\nEntities: "
                prompts.append(prompt)
        
        elif task == "Sentiment Analysis":
            for item in inputs:
                prompt = f"Analyze the sentiment of the following text:\n{item['text']}\nSentiment: "
                prompts.append(prompt)
        
        return prompts