from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gradio as gr

def initialize_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

chatbot = initialize_model()

def chat(input_text):
    response = chatbot(input_text, max_new_tokens=100, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

# Create Gradio interface
interface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="NagarTiran AI Chatbot",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size="md",
        text_size="md"
    )
)

# Launch the app
if __name__ == "__main__":
    interface.launch(share=True) 