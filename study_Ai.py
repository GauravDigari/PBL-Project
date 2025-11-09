from tkinter import *
from tkinter import scrolledtext
import threading
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import pyttsx3

# ------------------ Voice Setup ------------------
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('voice', engine.getProperty('voices')[0].id)

def speak(text):
    """Make the bot speak the reply aloud."""
    engine.say(text)
    engine.runAndWait()

# ------------------ Global Variables ------------------
current_subject = "default"
available_subjects = ["default", "daa", "java", "python", "dbms", "ai"]

# ------------------ GUI Setup ------------------
root = Tk()
root.title("PERSON STUDY ASSESSMENT CHATBOT")
root.geometry("1350x692")
root.config(bg="#F5F5F5")

# ------------------ Load Knowledge Base ------------------
def load_subject(subject):
    """Load the correct subject JSON file."""
    filename = f"{subject}.json"
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

knowledge = load_subject("default")  # default subject on startup

# ------------------ Chatbot Logic ------------------
def chatbot_response(user_input):
    global current_subject, knowledge

    # Check if user wants to switch subject
    for sub in available_subjects:
        if sub != "default" and sub in user_input.lower():
            current_subject = sub
            knowledge = load_subject(sub)
            return f"✅ Switched to {sub.upper()} mode! You can now ask me {sub} questions."

    # Normal Q&A
    if not knowledge:
        return "Sorry, I don't have information for this subject yet."

    questions = []
    responses = []
    for item in knowledge:
        for q in item["questions"]:
            questions.append(q)
            responses.append(item)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    item = responses[index]
    return item["answer"]

# ------------------ Send Button Logic ------------------
def on_send():
    user_input_text = Input_text.get(1.0, END).strip()
    if not user_input_text:
        return

    Display_text.config(state=NORMAL)
    Display_text.insert(END, f'You : {user_input_text}\n', "User")

    bot_reply = chatbot_response(user_input_text)
    Display_text.insert(END, f'Alpha: {bot_reply}\n\n', "Alpha")
    speak(bot_reply)

    Display_text.tag_config("User", foreground="#FF9800")
    Display_text.tag_config("Alpha", foreground="#3F51B5")
    Display_text.config(state=DISABLED)
    Display_text.see(END)
    Input_text.delete(1.0, END)

# ------------------ Voice Input Logic ------------------
def listen_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        Display_text.config(state=NORMAL)
        Display_text.insert(END, "🎤 Listening...\n", "Alpha")
        Display_text.config(state=DISABLED)
        Display_text.update()
        audio = r.listen(source)

    try:
        user_input_text = r.recognize_google(audio)
        Display_text.config(state=NORMAL)
        Display_text.insert(END, f'You (voice): {user_input_text}\n', "User")

        bot_reply = chatbot_response(user_input_text)
        Display_text.insert(END, f'Alpha: {bot_reply}\n\n', "Alpha")
        speak(bot_reply)

        Display_text.config(state=DISABLED)
        Display_text.see(END)
    except sr.UnknownValueError:
        Display_text.config(state=NORMAL)
        Display_text.insert(END, "Alpha: Sorry, I couldn't understand you.\n\n", "Alpha")
        speak("Sorry, I couldn't understand you.")
        Display_text.config(state=DISABLED)
    except sr.RequestError:
        Display_text.config(state=NORMAL)
        Display_text.insert(END, "Alpha: Network error. Try again later.\n\n", "Alpha")
        speak("Network error. Try again later.")
        Display_text.config(state=DISABLED)

# ------------------ Header ------------------
header = Label(
    text='PERSON STUDY ASSESSMENT CHATBOT',
    bg='#3F51B5',
    fg='white',
    font=('Arial', 30, 'bold')
)
header.pack(fill=BOTH, expand=True)

# ------------------ Chat Display ------------------
Display_text = scrolledtext.ScrolledText(
    root,
    wrap=WORD,
    state=DISABLED,
    bg='#FFFFFF',
    fg='#000000'
)
Display_text.pack(fill=BOTH, expand=True)

# ------------------ Input Area ------------------
Input_text = scrolledtext.ScrolledText(
    root,
    wrap=WORD,
    height=10,
    bg='#F5F5F5',
    fg='#000000',
    insertbackground='black'
)
Input_text.pack(fill=BOTH, expand=True)

# ---- Button Frame ----
button_frame = Frame(root, bg="#F5F5F5")
button_frame.pack(fill=X, pady=10)

# Mic button (Left)
mic_btn = Button(
    button_frame,
    text='🎙️ Speak',
    font=('Arial', 16, 'bold'),
    bg='#4CAF50',
    fg='white',
    activebackground='#45A049',
    activeforeground='white',
    command=listen_voice,
    width=10
)
mic_btn.pack(side=LEFT, padx=20)

# Send button (Right)
submit = Button(
    button_frame,
    text='SEND',
    font=('Arial', 16, 'bold'),
    bg='#FF9800',
    fg='white',
    activebackground='#F57C00',
    activeforeground='white',
    command=on_send,
    width=10
)
submit.pack(side=RIGHT, padx=20)


# ------------------ Enter Key Binding ------------------
def on_enter(event):
    on_send()
    return "break"

Input_text.bind("<Return>", on_enter)

# ------------------ Run App ------------------
root.mainloop()
