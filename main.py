import json
import random
import time
import os
from gtts import gTTS
import speech_recognition as sr
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Define the template
template = """
Answer the question
Here is the conversation history: {context}
Question: {question}
Answer concisely and do not ask another question in response:
"""

# Initialize the model and prompt template
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Load predefined questions from JSON file
with open('questions.json', 'r') as f:
    predefined_questions = json.load(f)

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")  # For Windows; use "afplay response.mp3" on macOS or "mpg123 response.mp3" on Linux

def handle_conversation():
    context = ""
    use_predefined_question = True
    count = 0

    print("Welcome to the AI chatBot! Say 'exit' to quit.")

    recognizer = sr.Recognizer()

    while True:
        # Ask a random predefined question or generate one
        if use_predefined_question:
            question = random.choice(predefined_questions)['question']
            use_predefined_question = False
        else:
            # Let the bot generate the next question
            result = chain.invoke({
                "context": context,
                "question": "What should I ask next? Please provide a single, concise response without asking another question. Just replace the words from the question: they, their to you, your."
            })
            question = result.strip().split('\n')[0]  # Ensure only the first line is taken as the question

        # Print the question only if it does not end with a '?'
        if (question.endswith('?')) or (count == 0):
            print(f"Bot: {question}")
            speak(question)
            time.sleep(8)  # Increased delay to ensure the audio plays before listening
            count += 1

        # Capture user input via voice
        with sr.Microphone() as source:
            print("Listening for your response...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)  # Adjust the timeout and phrase_time_limit as needed

        try:
            user_input = recognizer.recognize_google(audio)
            print(f"You: {user_input}")
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio. Please try again.")
            continue
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            continue

        if user_input.lower() == "exit":
            break

        # Generate response
        result = chain.invoke({
            "context": context,
            "question": user_input,
            "max_tokens": 50
        })
        response = result[:150]  # Limit the response display to 150 characters
        
        # Text-to-speech for the response
        speak(response)
        time.sleep(8)  # Increased delay to ensure the audio plays before continuing
        
        # Print response if it is not a question
        if not response.endswith('?'):
            print("Bot:", response)
        
        # Update context
        context += f"\nUser: {user_input}\nAI: {response}"

        # Prepare for the next question
        use_predefined_question = False

if __name__ == "__main__":
    handle_conversation()
