from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TextGenerator:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prompt = (
            "As you create the summary, follow these principles consistently to ensure accuracy and coherence:\n\n"
            "1. Detail and Complexity: Focus on crafting a summary that is in-depth and thorough, ensuring that all essential points from the text are covered comprehensively and also capture the numerical information from the article.\n"
            "2. Clarity and Conciseness: Maintain a consistent writing style that is clear and concise. Regularly check to eliminate any unnecessary words that don't contribute to the core message.\n"
            "3. Relevance: Consistently refer back to the provided text, ensuring that no external information is included in the summary. This keeps the summary aligned with the source material.\n"
            "4. Format: Consistently format the summary as a single, cohesive paragraph, making it easy to read and understand.\n"
            "5. Word Limit: Throughout the process, consistently check that the summary remains within the 85-word limit, ensuring it is both comprehensive and succinct.\n"
            "6. No Abrupt Ending: Summary should not have any abrupt ending; it must be completed, not leave the summary in between without completing it.\n"
            "7. Best Summary Selection: Generate multiple summaries for each article, and in the output, provide only the best summary that adheres to all the principles mentioned above.\n\n"
            "Only provide the summary without including phrases like 'Summary:', special characters, or introductory statements like 'Following the provided text, the best summary within the 80-word limit is:'.\n"
            "The content is provided below please work on it according to the above instructions:\n"
        )

    def generate_text(self, text: str, model, tokenizer):
        # Combine the prompt with the user-provided text and special end-of-text token
        full_prompt = self.prompt + "\n\n" + text + "<|endoftext|>"

       
        # # Generate text
        # outputs = model.generate(**input_ids, max_new_tokens=1024)
        # Generate text directly using the raw input prompt
        outputs = model.generate(full_prompt)

    # Extract the generated text from the CompletionOutput
        generated_text = outputs[0].outputs[0].text  # Access the first output's text

    # Optionally, remove the prompt from the generated text
        clean_text = generated_text.replace(full_prompt, "").strip()

        return clean_text
        
