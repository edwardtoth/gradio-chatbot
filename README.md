# Sentiment Chatbot

This project is a sentiment-based chatbot that uses OpenAI's GPT-3.5-turbo model to interact with users. The chatbot maintains a memory of past conversations and uses sentiment analysis and keyword extraction to provide more relevant responses.

## Features

- Sentiment analysis using NLTK's VADER
- Keyword extraction using RAKE
- Sentence embeddings using Sentence Transformers
- Memory management for past conversations
- Cosine similarity for finding relevant past conversations
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
    python mort-gpt.py
    ```

2. Open the Gradio interface in your browser and start interacting with the chatbot.

## Project Structure

- `mort-gpt.py`: Main script for the chatbot.
- `utils.py`: Utility functions for loading memory, finding relevant conversations, and summarizing text.
- `memory_manager.py`: Functions for updating and saving memory.
- `requirements.txt`: List of required Python packages.

## Dependencies

- `datetime`
- `json`
- `math`
- `os`
- `numpy`
- `nltk`
- `rake-nltk`
- `sentence-transformers`
- `scikit-learn`
- `python-dotenv`
- `openai`
- `gradio`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.# gradio-chatbot
