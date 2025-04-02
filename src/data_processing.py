# src/tasks/qa_processing.py

from transformers import pipeline

def process_qa_in_english(question, context):
    # Load a pre-trained QA model
    qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    # Process the QA task
    result = qa_pipeline(question=question, context=context)
    return result['answer']

if __name__ == "__main__":
    # Example QA processing
    question = "What is the capital of France?"
    context = "France is a country in Europe. The capital of France is Paris."
    answer = process_qa_in_english(question, context)
    print(f"Answer: {answer}")