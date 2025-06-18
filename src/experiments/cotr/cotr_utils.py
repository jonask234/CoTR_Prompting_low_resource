import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CoTRSystem:
    def __init__(self, model_name: str, use_auth_token: bool = False):
        """
        Initialize the CoTR system with a model and tokenizer.
        
        Args:
            model_name: The name of the Hugging Face model to use.
            use_auth_token: Whether to use an authentication token for private models.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_auth_token=use_auth_token
        )

    def process(self, text: str, **kwargs):
        """
        Process a given text through the CoTR pipeline.
        This method should be implemented by subclasses.
        
        Args:
            text: The input text to process.
            **kwargs: Additional arguments for processing.
            
        Returns:
            The processed output.
        """
        raise NotImplementedError("Each CoTR system must implement its own process method.")

    def create_prompt(self, text: str, **kwargs) -> str:
        """
        Create a prompt for the given task.
        This method should be implemented by subclasses.
        
        Args:
            text: The input text for which to create a prompt.
            **kwargs: Additional arguments for prompt creation.
            
        Returns:
            The formatted prompt.
        """
        raise NotImplementedError("Each CoTR system must implement its own create_prompt method.")

    def parse_response(self, response: str, **kwargs):
        """
        Parse the model's response to extract the desired information.
        This method should be implemented by subclasses.
        
        Args:
            response: The model's response.
            **kwargs: Additional arguments for parsing.
            
        Returns:
            The parsed information.
        """
        raise NotImplementedError("Each CoTR system must implement its own parse_response method.") 