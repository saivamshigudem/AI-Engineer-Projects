# ============================================================
# DAY 51/100
# PROJECT : CHATGPT CHATBOT
# PART 1 : CHATBOT FOUNDATION
#
# HARDWARE:
# - SLOW CPU
# - VERY LIMITED STORAGE
# - NO GPU REQUIRED
# ============================================================


# ============================================================
# SECTION 1 : IMPORT LIBRARIES
# ============================================================

import re
import random

print("Libraries Imported Successfully")


# ============================================================
# SECTION 2 : PROJECT CONFIGURATION
# ============================================================

BOT_NAME = "MiniGPT"

MAX_HISTORY = 10


print()
print("=" * 60)
print("CHATGPT CHATBOT")
print("=" * 60)

print("Bot Name     :", BOT_NAME)
print("Max History  :", MAX_HISTORY)
print("Device       : CPU")
print("Storage      : Minimal")


# ============================================================
# SECTION 3 : TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert text to lowercase

    text = text.lower()


    # Remove extra spaces

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    # Remove spaces from beginning and end

    text = text.strip()


    return text


# ============================================================
# SECTION 4 : LIGHTWEIGHT RESPONSE ENGINE
# ============================================================

def generate_response(user_message):

    # Clean user message

    message = clean_text(
        user_message
    )


    # --------------------------------------------------------
    # GREETINGS
    # --------------------------------------------------------

    if message in [
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there"
    ]:

        return random.choice([

            "Hello! How can I help you?",

            "Hi! What would you like to know?",

            "Hey! How can I assist you today?"

        ])


    # --------------------------------------------------------
    # HOW ARE YOU
    # --------------------------------------------------------

    if "how are you" in message:

        return (
            "I'm doing well! "
            "I'm ready to help you."
        )


    # --------------------------------------------------------
    # BOT IDENTITY
    # --------------------------------------------------------

    if (
        "your name" in message
        or
        "who are you" in message
    ):

        return (
            f"I'm {BOT_NAME}, "
            "a lightweight ChatGPT-style chatbot."
        )


    # --------------------------------------------------------
    # THANK YOU
    # --------------------------------------------------------

    if (
        "thank you" in message
        or
        "thanks" in message
    ):

        return random.choice([

            "You're welcome!",

            "Happy to help!",

            "You're welcome. Let me know if you need anything else."

        ])


    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    if (
        "what is ai" in message
        or
        "what is artificial intelligence" in message
    ):

        return (
            "Artificial Intelligence (AI) is the field "
            "of building systems that can perform tasks "
            "that normally require human intelligence."
        )


    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    if (
        "machine learning" in message
        or
        "what is ml" in message
    ):

        return (
            "Machine Learning is a branch of AI where "
            "models learn patterns from data and use "
            "those patterns to make predictions."
        )


    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    if "deep learning" in message:

        return (
            "Deep Learning is a type of Machine Learning "
            "that uses neural networks with multiple layers "
            "to learn complex patterns."
        )


    # --------------------------------------------------------
    # PYTHON
    # --------------------------------------------------------

    if "python" in message:

        return (
            "Python is a high-level programming language "
            "widely used for AI, Machine Learning, data "
            "science, automation, and backend development."
        )


    # --------------------------------------------------------
    # DEFAULT RESPONSE
    # --------------------------------------------------------

    return (
        "I understand your message: "
        f"'{user_message}'. "
        "I'm still a lightweight chatbot prototype, "
        "so my capabilities are limited."
    )


# ============================================================
# SECTION 5 : CHATBOT CLASS
# ============================================================

class ChatBot:

    def __init__(self):

        # Conversation memory

        self.history = []


    # --------------------------------------------------------
    # ADD MESSAGE
    # --------------------------------------------------------

    def add_message(
        self,
        role,
        content
    ):

        self.history.append({

            "role": role,

            "content": content

        })


    # --------------------------------------------------------
    # GENERATE CHAT RESPONSE
    # --------------------------------------------------------

    def chat(
        self,
        user_message
    ):

        # Store user message

        self.add_message(
            "user",
            user_message
        )


        # Generate response

        response = generate_response(
            user_message
        )


        # Store assistant response

        self.add_message(
            "assistant",
            response
        )


        # Keep only recent messages

        if len(self.history) > MAX_HISTORY:

            self.history = self.history[
                -MAX_HISTORY:
            ]


        return response


    # --------------------------------------------------------
    # SHOW CONVERSATION
    # --------------------------------------------------------

    def show_history(self):

        print()
        print("=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)


        if len(self.history) == 0:

            print("No conversation yet.")

            return


        for message in self.history:

            role = message["role"]

            content = message["content"]


            print(
                role.upper()
                + " : "
                + content
            )


    # --------------------------------------------------------
    # CLEAR CONVERSATION
    # --------------------------------------------------------

    def clear_history(self):

        self.history = []

        print(
            "Conversation history cleared."
        )


# ============================================================
# SECTION 6 : CREATE CHATBOT
# ============================================================

chatbot = ChatBot()


print()
print("=" * 60)
print("CHATBOT CREATED SUCCESSFULLY")
print("=" * 60)

print(
    "Bot:",
    BOT_NAME
)


# ============================================================
# SECTION 7 : TEST CHATBOT
# ============================================================

print()
print("=" * 60)
print("BASIC CHATBOT TEST")
print("=" * 60)


questions = [

    "Hello",

    "What is AI?",

    "What is machine learning?",

    "What is deep learning?",

    "What is Python?"

]


for question in questions:

    response = chatbot.chat(
        question
    )


    print()
    print("User:")
    print(question)


    print()
    print(BOT_NAME + ":")

    print(response)


# ============================================================
# SECTION 8 : DISPLAY CONVERSATION
# ============================================================

chatbot.show_history()


# ============================================================
# SECTION 9 : TEST CLEAR HISTORY
# ============================================================

print()
print("=" * 60)
print("TESTING MEMORY")
print("=" * 60)

print(
    "Messages stored:",
    len(chatbot.history)
)


# ============================================================
# SECTION 10 : PART 1 SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
✔ Chatbot foundation created
✔ Text preprocessing created
✔ Response engine created
✔ Conversation memory created
✔ Chatbot class created
✔ History management implemented
✔ Clear history implemented
✔ CPU-friendly
✔ Minimal storage
✔ No large model downloaded
✔ No training required

READY FOR PART 2
""")
# ============================================================
# DAY 51/100
# PROJECT : CHATGPT CHATBOT
# PART 2 : PROMPT + CONVERSATION CONTEXT
#
# CONTINUES DIRECTLY FROM PART-1
# ============================================================


# ============================================================
# SECTION 1 : SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are MiniGPT, a helpful and concise AI assistant.

Instructions:
1. Understand the user's question.
2. Use the available conversation context.
3. Give clear and useful responses.
4. Avoid unnecessary repetition.
5. Keep answers simple and easy to understand.
"""


print("System Prompt Created Successfully")


# ============================================================
# SECTION 2 : CONTEXT CONFIGURATION
# ============================================================

MAX_CONTEXT_MESSAGES = 6


print()
print("=" * 60)
print("CONTEXT CONFIGURATION")
print("=" * 60)

print(
    "Maximum Context Messages:",
    MAX_CONTEXT_MESSAGES
)


# ============================================================
# SECTION 3 : GET RECENT CONVERSATION
# ============================================================

def get_recent_context():

    # Take only the most recent messages

    recent_messages = chatbot.history[
        -MAX_CONTEXT_MESSAGES:
    ]


    return recent_messages


# ============================================================
# SECTION 4 : FORMAT CONVERSATION
# ============================================================

def format_conversation():

    recent_messages = get_recent_context()


    formatted_messages = []


    for message in recent_messages:

        role = message["role"]

        content = message["content"]


        if role == "user":

            formatted_messages.append(
                "User: " + content
            )


        elif role == "assistant":

            formatted_messages.append(
                "Assistant: " + content
            )


    return "\n".join(
        formatted_messages
    )


# ============================================================
# SECTION 5 : BUILD CHATGPT-STYLE PROMPT
# ============================================================

def build_prompt(user_message):

    conversation = format_conversation()


    prompt = (

        SYSTEM_PROMPT

        + "\n\n"

        + "Conversation History:\n"

        + conversation

        + "\n\n"

        + "Current User Message:\n"

        + user_message

        + "\n\n"

        + "Assistant:"

    )


    return prompt


# ============================================================
# SECTION 6 : TEST CONTEXT
# ============================================================

print()
print("=" * 60)
print("CURRENT CONVERSATION CONTEXT")
print("=" * 60)


context = format_conversation()


if context:

    print(context)

else:

    print("No conversation context available yet.")


# ============================================================
# SECTION 7 : TEST PROMPT
# ============================================================

test_question = (
    "What is artificial intelligence?"
)


final_prompt = build_prompt(
    test_question
)


print()
print("=" * 60)
print("GENERATED PROMPT")
print("=" * 60)

print(final_prompt)


# ============================================================
# SECTION 8 : CONTEXT-AWARE RESPONSE FUNCTION
# ============================================================

def context_response(user_message):

    message = clean_text(
        user_message
    )


    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if message in [
        "hi",
        "hello",
        "hey"
    ]:

        return (
            "Hello! How can I help you today?"
        )


    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    if (
        "what is ai" in message
        or
        "what is artificial intelligence" in message
    ):

        return (
            "Artificial Intelligence is the field "
            "of creating systems that can perform "
            "tasks that normally require human "
            "intelligence."
        )


    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    if "machine learning" in message:

        return (
            "Machine Learning is a branch of AI where "
            "models learn patterns from data and use "
            "those patterns to make predictions."
        )


    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    if "deep learning" in message:

        return (
            "Deep Learning is a type of Machine Learning "
            "that uses neural networks with multiple "
            "layers to learn complex patterns."
        )


    # --------------------------------------------------------
    # PYTHON
    # --------------------------------------------------------

    if "python" in message:

        return (
            "Python is a high-level programming language "
            "widely used in AI, Machine Learning, "
            "data science, automation, and backend "
            "development."
        )


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (
        "I understand your question. "
        "I'm currently a lightweight chatbot prototype "
        "with a limited local knowledge base."
    )


# ============================================================
# SECTION 9 : CHAT WITH CONTEXT
# ============================================================

def chat_with_context(user_message):

    # Store user message

    chatbot.add_message(
        "user",
        user_message
    )


    # Build prompt

    prompt = build_prompt(
        user_message
    )


    # Generate response

    response = context_response(
        user_message
    )


    # Store assistant response

    chatbot.add_message(
        "assistant",
        response
    )


    # Limit total history

    if len(chatbot.history) > MAX_HISTORY:

        chatbot.history = chatbot.history[
            -MAX_HISTORY:
        ]


    return response


# ============================================================
# SECTION 10 : TEST CONTEXT-AWARE CHAT
# ============================================================

print()
print("=" * 60)
print("CONTEXT-AWARE CHAT TEST")
print("=" * 60)


question_1 = (
    "What is machine learning?"
)


answer_1 = chat_with_context(
    question_1
)


print()
print("User:")
print(question_1)


print()
print("MiniGPT:")
print(answer_1)


# ============================================================
# SECTION 11 : FOLLOW-UP QUESTION
# ============================================================

question_2 = (
    "What is deep learning?"
)


answer_2 = chat_with_context(
    question_2
)


print()
print("User:")
print(question_2)


print()
print("MiniGPT:")
print(answer_2)


# ============================================================
# SECTION 12 : DISPLAY UPDATED CONTEXT
# ============================================================

print()
print("=" * 60)
print("UPDATED CONVERSATION CONTEXT")
print("=" * 60)


print(
    format_conversation()
)


# ============================================================
# SECTION 13 : BUILD FOLLOW-UP PROMPT
# ============================================================

follow_up_question = (
    "How is it different from machine learning?"
)


follow_up_prompt = build_prompt(
    follow_up_question
)


print()
print("=" * 60)
print("FOLLOW-UP PROMPT")
print("=" * 60)

print(follow_up_prompt)


# ============================================================
# SECTION 14 : CONTEXT SIZE CHECK
# ============================================================

print()
print("=" * 60)
print("CONTEXT SIZE")
print("=" * 60)

print(
    "Messages in full history:",
    len(chatbot.history)
)


print(
    "Messages used for context:",
    len(get_recent_context())
)


# ============================================================
# SECTION 15 : PART 2 SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
✔ System prompt created
✔ Recent conversation context implemented
✔ Conversation formatting implemented
✔ ChatGPT-style prompt construction implemented
✔ Context-aware response function created
✔ Follow-up conversation tested
✔ Context window implemented
✔ No model downloaded
✔ No training required
✔ CPU-friendly
✔ Minimal storage required

READY FOR PART 3
""")
# ============================================================
# DAY 51/100
# PROJECT : CHATGPT CHATBOT
# PART 3 : RESPONSE GENERATION + CHAT MEMORY
#
# CONTINUES DIRECTLY FROM PART-1 AND PART-2
# ============================================================


# ============================================================
# SECTION 1 : LIGHTWEIGHT KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE = {

    "artificial intelligence": (
        "Artificial Intelligence (AI) is the field of "
        "building computer systems that can perform "
        "tasks that normally require human intelligence."
    ),

    "machine learning": (
        "Machine Learning is a branch of AI where "
        "models learn patterns from data and use "
        "those patterns to make predictions."
    ),

    "deep learning": (
        "Deep Learning is a type of Machine Learning "
        "that uses neural networks with multiple layers "
        "to learn complex patterns from data."
    ),

    "neural network": (
        "A neural network is a machine learning model "
        "made of interconnected layers that learn "
        "patterns from data."
    ),

    "cnn": (
        "CNN stands for Convolutional Neural Network. "
        "It is a deep learning architecture commonly "
        "used for image classification and computer vision."
    ),

    "transformer": (
        "A Transformer is a neural network architecture "
        "based on attention mechanisms. It is widely used "
        "in modern language and multimodal AI systems."
    ),

    "python": (
        "Python is a high-level programming language "
        "widely used for AI, machine learning, data science, "
        "automation, and backend development."
    ),

    "rag": (
        "RAG stands for Retrieval-Augmented Generation. "
        "It retrieves relevant information from an external "
        "knowledge source and provides it to a language model "
        "to generate a more grounded response."
    ),

    "lora": (
        "LoRA, or Low-Rank Adaptation, is a parameter-efficient "
        "fine-tuning technique that trains a small number of "
        "additional parameters instead of updating the entire model."
    )

}


print("Knowledge Base Created Successfully")


# ============================================================
# SECTION 2 : FIND RELEVANT KNOWLEDGE
# ============================================================

def find_knowledge(user_message):

    message = clean_text(
        user_message
    )


    # Search for matching topic

    for topic in KNOWLEDGE_BASE:

        if topic in message:

            return KNOWLEDGE_BASE[topic]


    return None


# ============================================================
# SECTION 3 : INTENT DETECTION
# ============================================================

def detect_intent(user_message):

    message = clean_text(
        user_message
    )


    # Greeting

    if message in [
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there"
    ]:

        return "greeting"


    # Goodbye

    if message in [
        "bye",
        "goodbye",
        "exit"
    ]:

        return "goodbye"


    # Thanks

    if (
        "thank you" in message
        or
        "thanks" in message
    ):

        return "thanks"


    # Identity

    if (
        "who are you" in message
        or
        "your name" in message
    ):

        return "identity"


    # Knowledge question

    if (
        "what is" in message
        or
        "what are" in message
        or
        "explain" in message
        or
        "define" in message
    ):

        return "knowledge"


    # Default

    return "general"


# ============================================================
# SECTION 4 : RESPONSE GENERATOR
# ============================================================

def generate_chat_response(
    user_message
):

    # Detect intent

    intent = detect_intent(
        user_message
    )


    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if intent == "greeting":

        return (
            "Hello! I'm MiniGPT. "
            "How can I help you today?"
        )


    # --------------------------------------------------------
    # GOODBYE
    # --------------------------------------------------------

    if intent == "goodbye":

        return (
            "Goodbye! Have a great day."
        )


    # --------------------------------------------------------
    # THANKS
    # --------------------------------------------------------

    if intent == "thanks":

        return (
            "You're welcome! "
            "I'm happy to help."
        )


    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    if intent == "identity":

        return (
            "I'm MiniGPT, a lightweight "
            "ChatGPT-style chatbot designed "
            "to run with minimal CPU and storage."
        )


    # --------------------------------------------------------
    # KNOWLEDGE
    # --------------------------------------------------------

    if intent == "knowledge":

        knowledge = find_knowledge(
            user_message
        )


        if knowledge is not None:

            return knowledge


    # --------------------------------------------------------
    # GENERAL RESPONSE
    # --------------------------------------------------------

    return (
        "I understand your question, but I don't "
        "currently have enough information in my "
        "local knowledge base to provide a detailed answer."
    )


# ============================================================
# SECTION 5 : CHAT FUNCTION WITH MEMORY
# ============================================================

def chat(user_message):

    # --------------------------------------------------------
    # CHECK EMPTY MESSAGE
    # --------------------------------------------------------

    if not user_message.strip():

        return (
            "Please enter a message."
        )


    # --------------------------------------------------------
    # BUILD CONTEXT BEFORE RESPONSE
    # --------------------------------------------------------

    context_before = format_conversation()


    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    response = generate_chat_response(
        user_message
    )


    # --------------------------------------------------------
    # STORE USER MESSAGE
    # --------------------------------------------------------

    chatbot.add_message(
        "user",
        user_message
    )


    # --------------------------------------------------------
    # STORE ASSISTANT RESPONSE
    # --------------------------------------------------------

    chatbot.add_message(
        "assistant",
        response
    )


    # --------------------------------------------------------
    # LIMIT HISTORY
    # --------------------------------------------------------

    if len(chatbot.history) > MAX_HISTORY:

        chatbot.history = chatbot.history[
            -MAX_HISTORY:
        ]


    return response


# ============================================================
# SECTION 6 : TEST RESPONSE GENERATION
# ============================================================

print()
print("=" * 60)
print("RESPONSE GENERATION TEST")
print("=" * 60)


test_questions = [

    "Hello",

    "What is artificial intelligence?",

    "What is machine learning?",

    "What is deep learning?",

    "What is a CNN?",

    "What is RAG?"

]


for question in test_questions:

    answer = chat(
        question
    )


    print()
    print("User:")
    print(question)

    print()
    print("MiniGPT:")
    print(answer)


# ============================================================
# SECTION 7 : CHECK CONVERSATION MEMORY
# ============================================================

print()
print("=" * 60)
print("CONVERSATION MEMORY")
print("=" * 60)


for message in chatbot.history:

    print(
        message["role"].upper()
        + " : "
        + message["content"]
    )


# ============================================================
# SECTION 8 : CONTEXT WINDOW TEST
# ============================================================

print()
print("=" * 60)
print("CURRENT CONTEXT WINDOW")
print("=" * 60)


recent_context = get_recent_context()


for message in recent_context:

    print(
        message["role"].upper()
        + " : "
        + message["content"]
    )


# ============================================================
# SECTION 9 : BUILD CONTEXT FOR A NEW QUESTION
# ============================================================

new_question = (
    "Can you explain deep learning?"
)


new_prompt = build_prompt(
    new_question
)


print()
print("=" * 60)
print("CONTEXT-AWARE PROMPT")
print("=" * 60)

print(new_prompt)


# ============================================================
# SECTION 10 : TEST FOLLOW-UP QUESTIONS
# ============================================================

print()
print("=" * 60)
print("FOLLOW-UP QUESTION TEST")
print("=" * 60)


question_1 = (
    "What is machine learning?"
)


answer_1 = chat(
    question_1
)


print()
print("User:", question_1)

print(
    "MiniGPT:",
    answer_1
)


question_2 = (
    "What is deep learning?"
)


answer_2 = chat(
    question_2
)


print()
print("User:", question_2)

print(
    "MiniGPT:",
    answer_2
)


# ============================================================
# SECTION 11 : CHAT MEMORY SIZE
# ============================================================

print()
print("=" * 60)
print("MEMORY INFORMATION")
print("=" * 60)

print(
    "Maximum History:",
    MAX_HISTORY
)

print(
    "Current Messages:",
    len(chatbot.history)
)

print(
    "Context Messages:",
    len(get_recent_context())
)


# ============================================================
# SECTION 12 : PART 3 SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART 3 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
✔ Lightweight knowledge base created
✔ Intent detection implemented
✔ Response generation implemented
✔ Conversation memory connected
✔ Context window implemented
✔ Follow-up questions tested
✔ Context-aware prompts generated
✔ No large model downloaded
✔ No model training required
✔ CPU-friendly
✔ Minimal storage required

READY FOR PART 4
""")
# ============================================================
# DAY 51/100
# PROJECT : CHATGPT CHATBOT
# PART 4 : COMPLETE CHATGPT-STYLE APPLICATION
#
# CONTINUES FROM PART-1, PART-2 AND PART-3
# ============================================================


# ============================================================
# SECTION 1 : IMPORT JSON
# ============================================================

import json

print("JSON Library Imported Successfully")


# ============================================================
# SECTION 2 : CHAT SESSION
# ============================================================

class ChatSession:

    def __init__(self, chatbot):

        self.chatbot = chatbot


    # --------------------------------------------------------
    # SEND MESSAGE
    # --------------------------------------------------------

    def send_message(self, user_message):

        response = chat(
            user_message
        )

        return response


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    def clear_chat(self):

        self.chatbot.clear_history()


    # --------------------------------------------------------
    # SHOW HISTORY
    # --------------------------------------------------------

    def show_history(self):

        print()
        print("=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)


        if len(self.chatbot.history) == 0:

            print("No conversation available.")

            return


        for message in self.chatbot.history:

            role = message["role"]

            content = message["content"]


            if role == "user":

                print()
                print("YOU:")
                print(content)


            else:

                print()
                print(BOT_NAME + ":")

                print(content)


# ============================================================
# SECTION 3 : CREATE CHAT SESSION
# ============================================================

session = ChatSession(
    chatbot
)


print()
print("=" * 60)
print("CHAT SESSION CREATED")
print("=" * 60)

print(
    "Bot:",
    BOT_NAME
)


# ============================================================
# SECTION 4 : SAVE CONVERSATION
# ============================================================

CHAT_FILE = "chat_history.json"


def save_conversation():

    with open(
        CHAT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chatbot.history,
            file,
            indent=4,
            ensure_ascii=False
        )


    print()
    print(
        "Conversation saved successfully."
    )

    print(
        "File:",
        CHAT_FILE
    )


# ============================================================
# SECTION 5 : LOAD CONVERSATION
# ============================================================

def load_conversation():

    try:

        with open(
            CHAT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            chatbot.history = json.load(
                file
            )


        print()
        print(
            "Conversation loaded successfully."
        )

        print(
            "Messages:",
            len(chatbot.history)
        )


    except FileNotFoundError:

        print()
        print(
            "No saved conversation found."
        )


    except json.JSONDecodeError:

        print()
        print(
            "Saved conversation file is invalid."
        )


# ============================================================
# SECTION 6 : CHAT STATISTICS
# ============================================================

def show_statistics():

    total_messages = len(
        chatbot.history
    )


    user_messages = 0

    assistant_messages = 0


    for message in chatbot.history:

        if message["role"] == "user":

            user_messages += 1


        elif message["role"] == "assistant":

            assistant_messages += 1


    print()
    print("=" * 60)
    print("CHAT STATISTICS")
    print("=" * 60)

    print(
        "Total Messages     :",
        total_messages
    )

    print(
        "User Messages      :",
        user_messages
    )

    print(
        "Assistant Messages :",
        assistant_messages
    )

    print(
        "Context Messages   :",
        len(get_recent_context())
    )


# ============================================================
# SECTION 7 : DISPLAY CURRENT CONTEXT
# ============================================================

def show_context():

    print()
    print("=" * 60)
    print("CURRENT CONTEXT")
    print("=" * 60)


    context = format_conversation()


    if context:

        print(context)

    else:

        print(
            "No context available."
        )


# ============================================================
# SECTION 8 : COMPLETE INTERACTIVE CHAT
# ============================================================

def run_chatbot():

    print()
    print("=" * 60)
    print("             MINI GPT CHATBOT")
    print("=" * 60)

    print()
    print(
        "Welcome to",
        BOT_NAME
    )

    print()
    print("Available commands:")
    print("  history  → Show conversation")
    print("  context  → Show current context")
    print("  stats    → Show statistics")
    print("  save     → Save conversation")
    print("  load     → Load conversation")
    print("  clear    → Clear conversation")
    print("  exit     → Exit chatbot")

    print()


    while True:

        # ----------------------------------------------------
        # USER INPUT
        # ----------------------------------------------------

        user_message = input(
            "You: "
        )


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "exit":

            print()

            print(
                BOT_NAME
                + ": Goodbye! 👋"
            )

            break


        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "history":

            session.show_history()

            continue


        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "context":

            show_context()

            continue


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "stats":

            show_statistics()

            continue


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "save":

            save_conversation()

            continue


        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "load":

            load_conversation()

            continue


        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "clear":

            session.clear_chat()

            print(
                "Conversation cleared."
            )

            continue


        # ----------------------------------------------------
        # EMPTY MESSAGE
        # ----------------------------------------------------

        if not user_message.strip():

            print(
                BOT_NAME
                + ": Please enter a message."
            )

            continue


        # ----------------------------------------------------
        # GENERATE RESPONSE
        # ----------------------------------------------------

        response = session.send_message(
            user_message
        )


        print()
        print(
            BOT_NAME
            + ":"
        )

        print(response)

        print()


# ============================================================
# SECTION 9 : DEMO CONVERSATION
# ============================================================

print()
print("=" * 60)
print("DEMO CONVERSATION")
print("=" * 60)


demo_questions = [

    "Hello",

    "What is artificial intelligence?",

    "What is machine learning?",

    "What is deep learning?",

    "What is RAG?"

]


for question in demo_questions:

    response = session.send_message(
        question
    )


    print()
    print("You:")
    print(question)


    print()
    print(BOT_NAME + ":")

    print(response)


# ============================================================
# SECTION 10 : SHOW HISTORY
# ============================================================

session.show_history()


# ============================================================
# SECTION 11 : SHOW STATISTICS
# ============================================================

show_statistics()


# ============================================================
# SECTION 12 : SAVE CONVERSATION
# ============================================================

save_conversation()


# ============================================================
# SECTION 13 : FINAL PROJECT SUMMARY
# ============================================================

print()
print("=" * 60)
print("DAY 51/100 — PROJECT COMPLETED")
print("=" * 60)

print("""
PROJECT:
ChatGPT-Style Chatbot

PART 1:
✔ Chatbot foundation
✔ Text preprocessing
✔ Response engine
✔ Conversation memory

PART 2:
✔ System prompt
✔ Context management
✔ Conversation formatting
✔ Prompt construction

PART 3:
✔ Knowledge base
✔ Intent detection
✔ Response generation
✔ Context-aware responses

PART 4:
✔ Interactive chatbot
✔ Conversation history
✔ Context viewer
✔ Save conversation
✔ Load conversation
✔ Clear conversation
✔ Chat statistics

HARDWARE:
✔ CPU friendly
✔ Minimal storage
✔ No GPU required
✔ No large model download
✔ No model training

DAY 51/100 COMPLETE ✔
""")


# ============================================================
# SECTION 14 : START THE CHATBOT
# ============================================================

# Run this line when you are ready to chat.

# run_chatbot()
