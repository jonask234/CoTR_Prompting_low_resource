# src/pipelines/baseline_pipeline.py

class BaselinePipeline:
    def __init__(self, model, tokenizer):
        """
        Initialize the baseline pipeline.
        
        Args:
            model: The language model to use.
            tokenizer: The tokenizer for the language model.
        """
        self.model = model
        self.tokenizer = tokenizer
        
    def process(self, inputs, task, language):
        """
        Process inputs directly with the model without translation.
        
        Args:
            inputs (dict): Input data for the task.
            task (str): The NLP task to perform.
            language (str): The language of the inputs.
            
        Returns:
            dict: Model outputs.
        """
        # Construct prompt based on task and language
        prompts = self._construct_prompts(inputs, task, language)
        
        # Generate outputs
        outputs = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(
                input_ids,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(output)
        
        return {
            "task": task,
            "language": language,
            "inputs": inputs,
            "outputs": outputs
        }
    
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
                prompt = f"Summarize the following text in {language}:\n{item['text']}\nSummary: "
                prompts.append(prompt)
        
        elif task == "NER":
            for item in inputs:
                prompt = f"Identify the named entities in the following text in {language}:\n{item['text']}\nEntities: "
                prompts.append(prompt)
        
        elif task == "Sentiment Analysis":
            for item in inputs:
                prompt = f"Analyze the sentiment of the following text in {language}:\n{item['text']}\nSentiment: "
                prompts.append(prompt)
        
        return prompts