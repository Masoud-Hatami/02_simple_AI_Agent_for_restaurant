from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import json
from langchain.chains import RetrievalQA
from gtts import gTTS
import speech_recognition as sr
import os
import tempfile
import pygame

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,
    max_tokens=250,
)

# Load the menu from a JSON file
def load_menu(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

menu = load_menu("menu.json")

# Load restaurant information from a text file
def load_restaurant_info(file_path):
    with open(file_path, "r") as file:
        return file.read()

restaurant_info = load_restaurant_info("restaurant_info.txt")

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_text(restaurant_info)

# Create embeddings and a vector store
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(texts, embedding)

# Convert menu to a string
menu_str = json.dumps(menu, indent=2).replace("{", "{{").replace("}", "}}")

# Set up the RetrievalQA chain for restaurant info
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
)
print("Welcome to our Restaurant.")
# Chat prompt template
system = f"""You are a friendly and professional assistant. 
    Your tasks are:
    1. Take their order by asking, and calculate the total bill.
    2. If the customer orders something not on the menu, politely inform them that the item is not available and don't show the menu.
    3. Thank the customer for ordering and provide any additional assistance if needed.
    
    Always start with: Welcome to our Italian Restaurant!
    This is our restaurant menu: {menu_str}. When the customer asks, please show it."""

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{customer_input}")  # User input
])

# Create a chain: Prompt -> LLM -> Output
chatbot_chain = chat_prompt_template | llm | StrOutputParser()

# Function to convert text to speech

def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        fp.close()  # Ensure the file is written and closed

        # Initialize pygame and play the audio
        pygame.mixer.init()
        pygame.mixer.music.load(fp.name)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Unload the music to release the file handle
        pygame.mixer.music.unload()

        # Delete the file after playing
        os.remove(fp.name)

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:  # Fixed typo here
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, that service is down.")
            return None
# Initialize conversation history
conversation_history = []

# Run the conversation in a loop
while True:
    user_input = speech_to_text()
    if user_input is None:
        continue
    

    if user_input.lower() in ["bye", "exit", "thank you"]:
        response = "Thank you for visiting us! Have a great day!"
        print(f"Chatbot: {response}")
        text_to_speech(response)
        break

    # Append user input to the conversation history
    conversation_history.append(f"You: {user_input}")

    # Check if the query is about the restaurant (Use RAG)
    if any(keyword in user_input.lower() for keyword in ["open", "location", "contact", "special"]):
        response = qa_chain.invoke({"query": user_input})
        response_text = response["result"]
    else:
        # Handle menu and orders
        response_text = chatbot_chain.invoke({"customer_input": user_input})
        # Handle menu and orders
        # Include the conversation history in the prompt
        history_str = "\n".join(conversation_history)
        full_prompt = f"{history_str}\nChatbot:"
        response_text = chatbot_chain.invoke({"customer_input": full_prompt})

    # Append chatbot response to the conversation history
    conversation_history.append(f"Chatbot: {response_text}")

    print(f"Chatbot: {response_text}")
    text_to_speech(response_text)
