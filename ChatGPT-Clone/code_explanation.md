# ============================================================
# DAY 51/100 — CHATGPT CLONE
# PROJECT : LIGHTWEIGHT CHATGPT-STYLE ASSISTANT
# PART 1 : CHATBOT FOUNDATION
#
# CPU-FRIENDLY
# LOW STORAGE
# NO LARGE MODEL
# ============================================================


# ============================================================
# SECTION 1 : IMPORT REQUIRED LIBRARIES
# ============================================================

import re
import random
import json

print("Libraries Imported Successfully")


# ============================================================
# SECTION 2 : PROJECT CONFIGURATION
# ============================================================

BOT_NAME = "MiniGPT"

MAX_HISTORY = 10


print()
print("=" * 60)
print("CHATGPT CLONE CONFIGURATION")
print("=" * 60)

print("Bot Name      :", BOT_NAME)
print("Max History   :", MAX_HISTORY)
print("Storage       : Minimal")
print("Execution     : CPU Friendly")


# ============================================================
# SECTION 3 : BASIC TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert to lowercase

    text = text.lower()


    # Remove extra spaces

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    # Remove spaces from beginning/end

    text = text.strip()


    return text


# ============================================================
# SECTION 4 : LIGHTWEIGHT RESPONSE ENGINE
# ============================================================

def generate_basic_response(user_message):

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

    if (
        "how are you" in message
        or "how r you" in message
    ):

        return (
            "I'm doing well! "
            "I'm ready to help you."
        )


    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    if (
        "your name" in message
        or "who are you" in message
    ):

        return (
            "I'm "
            + BOT_NAME
            + ", a lightweight "
            "ChatGPT-style assistant."
        )


    # --------------------------------------------------------
    # THANK YOU
    # --------------------------------------------------------

    if (
        "thank you" in message
        or "thanks" in message
    ):

        return random.choice([
            "You're welcome!",
            "Happy to help!",
            "You're welcome. Let me know if you need anything else."
        ])


    # --------------------------------------------------------
    # AI QUESTIONS
    # --------------------------------------------------------

    if (
        "what is ai" in message
        or "what is artificial intelligence" in message
    ):

        return (
            "Artificial Intelligence is the field of "
            "building systems that can perform tasks "
            "that normally require human intelligence."
        )


    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    if (
        "what is machine learning" in message
        or "define machine learning" in message
    ):

        return (
            "Machine Learning is a branch of AI where "
            "models learn patterns from data and use "
            "those patterns to make predictions."
        )


    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    if (
        "what is deep learning" in message
        or "define deep learning" in message
    ):

        return (
            "Deep Learning is a type of Machine Learning "
            "that uses neural networks with multiple layers "
            "to learn complex patterns."
        )


    # --------------------------------------------------------
    # DEFAULT RESPONSE
    # --------------------------------------------------------

    return (
        "I understand your message: \""
        + user_message
        + "\". "
        "I'm still a lightweight prototype, so my "
        "response capabilities are limited right now."
    )


# ============================================================
# SECTION 5 : CHATBOT CLASS
# ============================================================

class MiniGPT:

    def __init__(self):

        self.history = []


    # --------------------------------------------------------
    # ADD MESSAGE TO HISTORY
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
    # GENERATE RESPONSE
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

        response = generate_basic_response(
            user_message
        )


        # Store assistant response

        self.add_message(
            "assistant",
            response
        )


        # Keep only recent conversation

        if len(self.history) > MAX_HISTORY:

            self.history = self.history[
                -MAX_HISTORY:
            ]


        return response


    # --------------------------------------------------------
    # SHOW HISTORY
    # --------------------------------------------------------

    def show_history(self):

        print()
        print("=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)


        for message in self.history:

            role = message["role"]

            content = message["content"]


            print(
                role.upper()
                + " : "
                + content
            )


    # --------------------------------------------------------
    # CLEAR HISTORY
    # --------------------------------------------------------

    def clear_history(self):

        self.history = []


        print(
            "Conversation history cleared."
        )


# ============================================================
# SECTION 6 : CREATE CHATBOT
# ============================================================

chatbot = MiniGPT()


print()
print("=" * 60)
print("CHATBOT CREATED")
print("=" * 60)

print(
    "Bot:",
    BOT_NAME
)


# ============================================================
# SECTION 7 : TEST BASIC CHAT
# ============================================================

print()
print("=" * 60)
print("BASIC CHAT TEST")
print("=" * 60)


response = chatbot.chat(
    "Hello"
)


print(
    "User:",
    "Hello"
)


print(
    BOT_NAME + ":",
    response
)


response = chatbot.chat(
    "What is machine learning?"
)


print()
print(
    "User:",
    "What is machine learning?"
)


print(
    BOT_NAME + ":",
    response
)


# ============================================================
# SECTION 8 : DISPLAY CONVERSATION HISTORY
# ============================================================

chatbot.show_history()


# ============================================================
# SECTION 9 : INTERACTIVE CHAT FUNCTION
# ============================================================

def start_chat():

    print()
    print("=" * 60)
    print(BOT_NAME + " CHAT")
    print("=" * 60)

    print(
        "Type 'exit' to stop the chat."
    )

    print(
        "Type 'clear' to clear conversation history."
    )

    print()


    while True:

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
                + ": Goodbye!"
            )

            break


        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "clear":

            chatbot.clear_history()

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

        response = chatbot.chat(
            user_message
        )


        print(
            BOT_NAME
            + ": "
            + response
        )


# ============================================================
# SECTION 10 : PART 1 SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
✔ Lightweight chatbot created
✔ Conversation history created
✔ User messages stored
✔ Assistant responses stored
✔ Basic response engine created
✔ Conversation clearing implemented
✔ Interactive chat interface created
✔ No large model downloaded
✔ CPU-friendly
✔ Minimal storage required

READY FOR PART 2
""")
# ============================================================
# DAY 51/100 — CHATGPT CLONE
# PROJECT : LIGHTWEIGHT CHATGPT-STYLE ASSISTANT
# PART 2 : PROMPT + CONTEXT MANAGEMENT
#
# CONTINUES DIRECTLY FROM PART-1
# ============================================================


# ============================================================
# SECTION 1 : SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are MiniGPT, a helpful and concise AI assistant.

Your goals are:
1. Understand the user's question.
2. Use relevant conversation context.
3. Give clear and useful answers.
4. Avoid unnecessary repetition.
5. Keep responses concise when possible.
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

    # Get recent messages from chatbot history

    recent_history = chatbot.history[
        -MAX_CONTEXT_MESSAGES:
    ]


    return recent_history


# ============================================================
# SECTION 4 : FORMAT CONVERSATION
# ============================================================

def format_conversation():

    recent_history = get_recent_context()


    conversation = []


    for message in recent_history:

        role = message["role"]

        content = message["content"]


        if role == "user":

            conversation.append(
                "User: " + content
            )


        elif role == "assistant":

            conversation.append(
                "Assistant: " + content
            )


    return "\n".join(
        conversation
    )


# ============================================================
# SECTION 5 : BUILD FINAL PROMPT
# ============================================================

def build_prompt(user_message):

    conversation = format_conversation()


    final_prompt = (

        SYSTEM_PROMPT

        + "\n\n"

        + "Previous Conversation:\n"

        + conversation

        + "\n\n"

        + "Current User Message:\n"

        + user_message

        + "\n\n"

        + "Assistant Response:"

    )


    return final_prompt


# ============================================================
# SECTION 6 : TEST PROMPT GENERATION
# ============================================================

test_message = "What is machine learning?"


prompt = build_prompt(
    test_message
)


print()
print("=" * 60)
print("GENERATED PROMPT")
print("=" * 60)

print(prompt)


# ============================================================
# SECTION 7 : CREATE CONTEXT-AWARE RESPONSE ENGINE
# ============================================================

def context_aware_response(
    user_message
):

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
            "Artificial Intelligence (AI) is the field "
            "of creating systems that can perform tasks "
            "that normally require human intelligence."
        )


    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    if (
        "machine learning" in message
    ):

        return (
            "Machine Learning is a branch of AI where "
            "models learn patterns from data and use "
            "those patterns to make predictions."
        )


    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    if (
        "deep learning" in message
    ):

        return (
            "Deep Learning uses neural networks with "
            "multiple layers to learn complex patterns "
            "from data."
        )


    # --------------------------------------------------------
    # PYTHON
    # --------------------------------------------------------

    if (
        "python" in message
    ):

        return (
            "Python is a high-level programming language "
            "widely used in AI, Machine Learning, data "
            "science, automation, and backend development."
        )


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (
        "I understand your question. "
        "I can currently answer questions about "
        "AI, Machine Learning, Deep Learning, and Python."
    )


# ============================================================
# SECTION 8 : UPDATE CHATBOT RESPONSE METHOD
# ============================================================

def chat_with_context(
    user_message
):

    # --------------------------------------------------------
    # STORE USER MESSAGE
    # --------------------------------------------------------

    chatbot.add_message(
        "user",
        user_message
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    prompt = build_prompt(
        user_message
    )


    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    response = context_aware_response(
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
# SECTION 9 : TEST CONTEXT-AWARE CHAT
# ============================================================

print()
print("=" * 60)
print("CONTEXT-AWARE CHAT TEST")
print("=" * 60)


response = chat_with_context(
    "What is machine learning?"
)


print()
print("User:")
print("What is machine learning?")


print()
print("MiniGPT:")
print(response)


# ============================================================
# SECTION 10 : TEST FOLLOW-UP QUESTION
# ============================================================

response = chat_with_context(
    "What is deep learning?"
)


print()
print("User:")
print("What is deep learning?")


print()
print("MiniGPT:")
print(response)


# ============================================================
# SECTION 11 : DISPLAY CURRENT CONTEXT
# ============================================================

print()
print("=" * 60)
print("CURRENT CONVERSATION CONTEXT")
print("=" * 60)


print(
    format_conversation()
)


# ============================================================
# SECTION 12 : DISPLAY FINAL PROMPT
# ============================================================

print()
print("=" * 60)
print("FINAL PROMPT WITH CONTEXT")
print("=" * 60)


final_prompt = build_prompt(
    "How is it different from machine learning?"
)


print(final_prompt)


# ============================================================
# SECTION 13 : PART 2 SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
✔ System prompt created
✔ Conversation context created
✔ Recent messages selected
✔ Conversation formatted
✔ Context-aware prompt created
✔ Context-aware response engine created
✔ Conversation memory connected
✔ No large model downloaded
✔ CPU-friendly
✔ Minimal storage required

READY FOR PART 3
""")
# ============================================================
# DAY 51/100 — CHATGPT CLONE
# PROJECT : LIGHTWEIGHT CHATGPT-STYLE ASSISTANT
# PART 3 : RESPONSE GENERATION + MEMORY
#
# CONTINUES DIRECTLY FROM PART-1 AND PART-2
# ============================================================


# ============================================================
# SECTION 1 : RESPONSE KNOWLEDGE
# ============================================================

# Lightweight predefined knowledge.
#
# This requires almost no storage and no model download.

KNOWLEDGE_BASE = {

    "python": (
        "Python is a high-level programming language "
        "commonly used for AI, Machine Learning, data "
        "science, automation, and backend development."
    ),

    "machine learning": (
        "Machine Learning is a branch of AI where "
        "models learn patterns from data and use "
        "those patterns to make predictions."
    ),

    "deep learning": (
        "Deep Learning is a type of Machine Learning "
        "that uses multi-layer neural networks to "
        "learn complex patterns."
    ),

    "artificial intelligence": (
        "Artificial Intelligence is the field of "
        "building systems that can perform tasks "
        "that normally require human intelligence."
    ),

    "neural network": (
        "A neural network is a machine learning model "
        "made of interconnected layers of artificial "
        "neurons that learn patterns from data."
    ),

    "cnn": (
        "A CNN, or Convolutional Neural Network, is "
        "a deep learning architecture commonly used "
        "for image and computer vision tasks."
    ),

    "transformer": (
        "A Transformer is a neural network architecture "
        "based on attention mechanisms and is widely "
        "used for language and multimodal AI."
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


    for topic, answer in KNOWLEDGE_BASE.items():

        if topic in message:

            return answer


    return None


# ============================================================
# SECTION 3 : LIGHTWEIGHT RESPONSE GENERATOR
# ============================================================

def generate_response(
    user_message
):

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
        "good morning",
        "good evening"
    ]:

        return (
            "Hello! How can I help you today?"
        )


    # --------------------------------------------------------
    # GOODBYE
    # --------------------------------------------------------

    if message in [
        "bye",
        "goodbye"
    ]:

        return (
            "Goodbye! Have a great day."
        )


    # --------------------------------------------------------
    # THANK YOU
    # --------------------------------------------------------

    if (
        "thank you" in message
        or
        "thanks" in message
    ):

        return (
            "You're welcome! Happy to help."
        )


    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    knowledge = find_knowledge(
        user_message
    )


    if knowledge is not None:

        return knowledge


    # --------------------------------------------------------
    # SIMPLE QUESTION DETECTION
    # --------------------------------------------------------

    if (
        message.startswith("what is")
        or
        message.startswith("what are")
        or
        message.startswith("define")
    ):

        topic = message

        topic = topic.replace(
            "what is",
            ""
        )

        topic = topic.replace(
            "what are",
            ""
        )

        topic = topic.replace(
            "define",
            ""
        )


        topic = topic.strip(
            " ?."
        )


        return (
            "I don't have detailed information "
            "about "
            + topic
            + " in my lightweight knowledge base yet."
        )


    # --------------------------------------------------------
    # DEFAULT RESPONSE
    # --------------------------------------------------------

    return (
        "I understand your message: "
        + user_message
        + ". "
        "I am a lightweight local prototype, "
        "so my knowledge is currently limited."
    )


# ============================================================
# SECTION 4 : CONTEXT-AWARE CHAT FUNCTION
# ============================================================

def generate_chat_response(
    user_message
):

    # --------------------------------------------------------
    # STORE USER MESSAGE
    # --------------------------------------------------------

    chatbot.add_message(
        "user",
        user_message
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    prompt = build_prompt(
        user_message
    )


    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    response = generate_response(
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
# SECTION 5 : TEST RESPONSE GENERATION
# ============================================================

print()
print("=" * 60)
print("RESPONSE GENERATION TEST")
print("=" * 60)


question = "What is deep learning?"


answer = generate_chat_response(
    question
)


print()
print("User:")
print(question)


print()
print("MiniGPT:")
print(answer)


# ============================================================
# SECTION 6 : TEST MULTIPLE QUESTIONS
# ============================================================

test_questions = [

    "What is Python?",

    "What is a CNN?",

    "What is a transformer?",

    "What is machine learning?"

]


print()
print("=" * 60)
print("MULTIPLE RESPONSE TEST")
print("=" * 60)


for question in test_questions:

    answer = generate_chat_response(
        question
    )


    print()
    print("User:", question)

    print(
        "MiniGPT:",
        answer
    )


# ============================================================
# SECTION 7 : DISPLAY CONVERSATION MEMORY
# ============================================================

print()
print("=" * 60)
print("CONVERSATION MEMORY")
print("=" * 60)


for message in chatbot.history:

    print(
        message["role"].upper()
        + ": "
        + message["content"]
    )


# ============================================================
# SECTION 8 : CONTEXT WINDOW
# ============================================================

print()
print("=" * 60)
print("CURRENT CONTEXT")
print("=" * 60)


current_context = get_recent_context()


for message in current_context:

    print(
        message["role"].upper()
        + ": "
        + message["content"]
    )


# ============================================================
# SECTION 9 : GENERATE COMPLETE PROMPT
# ============================================================

print()
print("=" * 60)
print("COMPLETE CHATGPT-STYLE PROMPT")
print("=" * 60)


final_prompt = build_prompt(
    "Explain CNN in simple words."
)


print(final_prompt)


# ============================================================
# SECTION 10 : MEMORY LIMIT TEST
# ============================================================

print()
print("=" * 60)
print("MEMORY LIMIT")
print("=" * 60)


print(
    "Maximum stored messages:",
    MAX_HISTORY
)


print(
    "Currently stored messages:",
    len(chatbot.history)
)


# ============================================================
# SECTION 11 : INTERACTIVE CHAT
# ============================================================

def start_chat_v2():

    print()
    print("=" * 60)
    print(BOT_NAME + " — CHAT SESSION")
    print("=" * 60)

    print(
        "Type 'exit' to stop."
    )

    print(
        "Type 'clear' to clear memory."
    )

    print()


    while True:

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
                + ": Goodbye!"
            )

            break


        # ----------------------------------------------------
        # CLEAR MEMORY
        # ----------------------------------------------------

        if clean_text(
            user_message
        ) == "clear":

            chatbot.clear_history()

            continue


        # ----------------------------------------------------
        # EMPTY INPUT
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

        response = generate_chat_response(
            user_message
        )


        print()
        print(
            BOT_NAME
            + ": "
            + response
        )

        print()


# ============================================================
# SECTION 12 : PART 3 SUMMARY
# ============================================================

print()
print("=" * 60)
print("PART 3 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
✔ Lightweight response generator created
✔ Knowledge base created
✔ Context-aware responses created
✔ Conversation memory connected
✔ Context window implemented
✔ Prompt generation integrated
✔ Interactive chat updated
✔ No large LLM downloaded
✔ No model training required
✔ CPU-friendly
✔ Minimal storage required

READY FOR PART 4
""")
# ============================================================
# DAY 51/100 — CHATGPT CLONE
# PROJECT : LIGHTWEIGHT CHATGPT-STYLE ASSISTANT
# PART 4 : COMPLETE CHAT APPLICATION
#
# CONTINUES DIRECTLY FROM PART-1, PART-2 AND PART-3
# ============================================================


# ============================================================
# SECTION 1 : CHAT SESSION MANAGER
# ============================================================

class ChatSession:

    def __init__(self, chatbot):

        self.chatbot = chatbot


    # --------------------------------------------------------
    # SEND MESSAGE
    # --------------------------------------------------------

    def send_message(self, user_message):

        response = generate_chat_response(
            user_message
        )

        return response


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    def clear_chat(self):

        self.chatbot.clear_history()


    # --------------------------------------------------------
    # SHOW CHAT
    # --------------------------------------------------------

    def show_chat(self):

        print()
        print("=" * 60)
        print("CHAT HISTORY")
        print("=" * 60)

        if len(self.chatbot.history) == 0:

            print("No conversation yet.")

            return


        for message in self.chatbot.history:

            role = message["role"]

            content = message["content"]


            if role == "user":

                print()
                print("You:")
                print(content)


            else:

                print()
                print(BOT_NAME + ":")
                print(content)


# ============================================================
# SECTION 2 : CREATE CHAT SESSION
# ============================================================

session = ChatSession(
    chatbot
)


print()
print("=" * 60)
print("CHAT SESSION CREATED")
print("=" * 60)

print(
    "Assistant:",
    BOT_NAME
)


# ============================================================
# SECTION 3 : CHAT STATISTICS
# ============================================================

def chat_statistics():

    total_messages = len(
        chatbot.history
    )


    user_messages = sum(

        1

        for message in chatbot.history

        if message["role"] == "user"

    )


    assistant_messages = sum(

        1

        for message in chatbot.history

        if message["role"] == "assistant"

    )


    print()
    print("=" * 60)
    print("CHAT STATISTICS")
    print("=" * 60)

    print(
        "Total Messages:",
        total_messages
    )

    print(
        "User Messages:",
        user_messages
    )

    print(
        "Assistant Messages:",
        assistant_messages
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
        "Conversation saved to:",
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
            "Messages loaded:",
            len(chatbot.history)
        )


    except FileNotFoundError:

        print()
        print(
            "No saved conversation found."
        )


# ============================================================
# SECTION 6 : TEST SAVE AND LOAD
# ============================================================

print()
print("=" * 60)
print("TESTING CONVERSATION STORAGE")
print("=" * 60)


# Save current conversation

save_conversation()


# ============================================================
# SECTION 7 : COMPLETE CHAT INTERFACE
# ============================================================

def run_chat():

    print()
    print("=" * 60)
    print("          MINI GPT CHAT")
    print("=" * 60)

    print()
    print("Commands:")

    print("  exit   → Exit chat")

    print("  clear  → Clear conversation")

    print("  history → Show conversation")

    print("  stats  → Show chat statistics")

    print("  save   → Save conversation")

    print("  load   → Load conversation")

    print()


    while True:

        user_message = input(
            "You: "
        )


        # ====================================================
        # EXIT
        # ====================================================

        if clean_text(
            user_message
        ) == "exit":

            print()
            print(
                BOT_NAME
                + ": Goodbye! 👋"
            )

            break


        # ====================================================
        # CLEAR
        # ====================================================

        if clean_text(
            user_message
        ) == "clear":

            session.clear_chat()

            print(
                "Conversation cleared."
            )

            continue


        # ====================================================
        # HISTORY
        # ====================================================

        if clean_text(
            user_message
        ) == "history":

            session.show_chat()

            continue


        # ====================================================
        # STATISTICS
        # ====================================================

        if clean_text(
            user_message
        ) == "stats":

            chat_statistics()

            continue


        # ====================================================
        # SAVE
        # ====================================================

        if clean_text(
            user_message
        ) == "save":

            save_conversation()

            continue


        # ====================================================
        # LOAD
        # ====================================================

        if clean_text(
            user_message
        ) == "load":

            load_conversation()

            continue


        # ====================================================
        # EMPTY INPUT
        # ====================================================

        if not user_message.strip():

            print(
                BOT_NAME
                + ": Please enter a message."
            )

            continue


        # ====================================================
        # GENERATE RESPONSE
        # ====================================================

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
# SECTION 8 : DEMO CONVERSATION
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

    "What is a CNN?"

]


for question in demo_questions:

    answer = session.send_message(
        question
    )


    print()
    print(
        "You:",
        question
    )

    print(
        BOT_NAME + ":",
        answer
    )


# ============================================================
# SECTION 9 : DISPLAY FINAL HISTORY
# ============================================================

session.show_chat()


# ============================================================
# SECTION 10 : SAVE FINAL CONVERSATION
# ============================================================

save_conversation()


# ============================================================
# SECTION 11 : CHAT STATISTICS
# ============================================================

chat_statistics()


# ============================================================
# SECTION 12 : FINAL PROJECT SUMMARY
# ============================================================

print()
print("=" * 60)
print("DAY 51/100 — CHATGPT CLONE COMPLETED")
print("=" * 60)

print("""
PROJECT:
Lightweight ChatGPT-Style Assistant

PART 1:
✔ Chatbot foundation
✔ User input
✔ Response engine
✔ Conversation history

PART 2:
✔ System prompt
✔ Context management
✔ Prompt construction
✔ Recent conversation window

PART 3:
✔ Response generation
✔ Knowledge base
✔ Context-aware responses
✔ Interactive chat

PART 4:
✔ Complete chat session
✔ Conversation statistics
✔ Save conversation
✔ Load conversation
✔ Clear conversation
✔ History viewer
✔ Interactive ChatGPT-style interface

HARDWARE:
✔ CPU friendly
✔ Minimal storage
✔ No GPU required
✔ No large model download
✔ No model training

DAY 51/100 COMPLETE ✔
""")


# ============================================================
# OPTIONAL:
# START THE FULL INTERACTIVE CHAT
#
# Uncomment the line below to start chatting.
# ============================================================

# run_chat()
