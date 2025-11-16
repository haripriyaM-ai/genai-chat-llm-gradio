# EXP-08 : Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework
## NAME : HARI PRIYA M
## REGISTER NO : 212224240047

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:
In many applications, users need a simple way to interact with AI models without writing code, but setting up a chatbot interface from scratch is often difficult because it involves handling APIs, managing conversation history, and designing a user-friendly UI. To solve this problem, we build a chatbot using the Falcon model and a Gradio interface so that users can easily send messages, get responses, and experience an interactive AI system directly in the browser. This project removes the complexity of backend handling and provides an accessible environment for real-time conversational AI.

### DESIGN STEPS:

<b>STEP 1: </b>Load environment variables and API keys required for connecting to the Falcon model.

<b>STEP 2: </b>Initialize the text-generation client and prepare a function to format user and bot messages into a proper prompt.

<b>STEP 3: </b>Create a respond function that sends the formatted prompt to the model and updates the conversation history.

<b>STEP 4: </b>Build the Gradio interface with chatbot, textbox, submit button, and clearing options.

<b>STEP 5: </b>Launch the app and allow real-time interaction with the AI model through the browser.


### PROGRAM:

```python
import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
```

```python
# Helper function
import requests, json
from text_generation import Client

#FalcomLM-instruct endpoint on the text_generation library
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)
```

```python
import gradio as gr
def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
        formatted_prompt = format_chat_prompt(message, chat_history)
        bot_message = client.generate(formatted_prompt,
                                     max_new_tokens=1024,
                                     stop_sequences=["\nUser:", "<|endoftext|>"]).generated_text
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT3']))
```

### OUTPUT:
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/e83ab16c-4fe5-45ad-94e7-051502d5640c" />

### RESULT:
The chatbot system was successfully developed and launched using FalconLM and Gradio. It accepts user queries, communicates with the model through the API, and displays accurate responses while maintaining full chat history.
