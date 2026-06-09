import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

SYSTEM_MESSAGE = (
    "You are a friendly and knowledgeable travel advisor specializing in "
    "budget-friendly trips. When a user asks about a destination, suggest "
    "the best time to visit, one must-see attraction, and one money-saving tip. "
    "Keep responses under 80 words. Example format:\n"
    "🗓 Best time: [season/month]\n"
    "📍 Must-see: [attraction]\n"
    "💰 Budget tip: [tip]"
)

def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = ""

    for chunk in client.chat_completion(
        messages,
        max_tokens = 256,
        temperature = 0.7,
        top_p = 0.9,
        stream = True,
    ):
        token = chunk.choices[0].delta.content
        response += token
        yield response

chatbot = gr.ChatInterface(respond)
chatbot.launch(debug = True)