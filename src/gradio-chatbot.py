import gradio as gr
import os
import dotenv
from openai import OpenAI
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

def respond(message, history: list[dict]):
    message = message + "\n" + 'Be friendly and brief. Max 2 sentences.'
    conversation_history = history + [{"role": "user", "content": message}]
    print(conversation_history)
    topic, sentiment = score_topic_and_sentiment(message)
    response = {"role": "assistant", "content": "🚀: "}
    for message in generate_assistant_response(conversation_history):
        token = message.choices[0].delta.content
        if token is not None:
            response['content'] += token
        
    # Append the formatted topic to the response content
    response['content'] += f"\n\n{topic}"
    response['content'] += f"\n{sentiment}"
    
    yield response

def generate_assistant_response(conversation_history, stream= True):
    """
    Generate the assistant's response using OpenAI's ChatCompletion API.
    """
    # Load the OpenAI API key from the .env file
    dotenv.load_dotenv()
    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response =client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=conversation_history,
        max_tokens=50,
        stream=stream,
        temperature=0.7,
        top_p=0.95,
    )
    return response


def score_topic_and_sentiment(text):
    """
    Initialize the tokenizer, model, and pipeline for text classification,
    """
    intent_classifier = IntentClassifier()
    prompt = build_prompt(text = text, company_name = "PayPal")
    topic = intent_classifier.predict(prompt)
    topic = 'Intent: ' + topic.title()
    pipe = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")
    sentiment = pipe(f'This is the topic {topic} for {text}')
    sentiment = 'User Sentiment' + format_for_display(sentiment)
    return topic, sentiment


def format_for_display(topic):
    """
    Format the topic classification result for display in the UI.
    """
    print(topic)
    label = topic[0]['label']
    score = topic[0]['score']
    formatted_score = round(score * 100, 1)
    return f": {label}, Confidence: {formatted_score}%"


class IntentClassifier:
    def __init__(self):
        # Load fine-tuned model by HuggingFace Model Hub
        HUGGINGFACE_MODEL_PATH = "Serj/intent-classifier"
        self.pipe = pipeline("text2text-generation", model=HUGGINGFACE_MODEL_PATH)


    def predict(self, input_text) -> str:
        print('Input:\n', input_text)
        output = self.pipe(input_text)
        output_text = output[0]['generated_text']
        print('Prediction:\n', output_text)
        return output_text

def build_prompt(text, company_name="",prompt_options="",  company_specific=""):
    """
    Build the chatbot prompt with the company name and specific information.
    """
    company_name = "FinTech"
    company_specific = "secure online payment solutions for individuals and businesses."
    prompt_options =  "OPTIONS:\n Password\n refund\n cancel subscription\n damaged item\n return item\n track order\n change address\n update payment method\n technical support\n product inquiry\n other\n" 
    return f"Company name: {company_name} is doing: {company_specific}\nCustomer: {text}.\nEND MESSAGE\nChoose one topic that matches customer's issue.\n{prompt_options}\nClass name: "

def test_interface():
    """
    Test the chatbot interface with a predefined message.
    """
    test_message = "I really want to learn Spanish"
    test_message = build_prompt(text = "I want to cancel subscription.")
    history = []
    for response in respond(test_message, history):
        print(response)
        

if __name__ == "__main__":
    m = IntentClassifier()
    input_text = build_prompt(text = "I want to cancel subscription.")
    output_text = m.predict(input_text)
    # Test the chatbot interface with a predefined message
    test_interface()
    # Launch the Gradio chatbot interface
    demo = gr.ChatInterface(respond, type="messages")
    demo.launch()
