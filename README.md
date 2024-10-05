# Sentiment Chatbot

This project is a sentiment-based chatbot that uses OpenAI's GPT-3.5-turbo model to interact with users. The chatbot provides an intent and sentiment score after the prompt.

## Features

- Intent score using [pre-trained classifier](#https://huggingface.co/Serj/intent-classifier)
- Sentiment analysis using [Roberta text generation model](#https://huggingface.co/siebert/sentiment-roberta-large-english)
- Gradio interface for easy interaction

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/sentiment-chatbot.git
    cd sentiment-chatbot
    ```

2. Create a virtual environment and activate it:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Run the chatbot:

    ```sh
    python gradio-chatbot.py
    ```


This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.# gradio-chatbot
