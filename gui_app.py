"""
GUI Desktop Application for Live NLP Chatbot
Built with Tkinter for Final Year Project Demonstration
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from chatbot import NLPChatbot

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live AI Chatbot - Final Year Project")
        self.root.geometry("520x650")
        self.root.minsize(450, 550)
        self.root.configure(bg="#F3F4F6")

        # Initialize the NLP Chatbot backend
        self.bot = None
        self.init_backend()

        self.setup_ui()

    def init_backend(self):
        try:
            self.bot = NLPChatbot(corpus_path="corpus.txt")
        except Exception as e:
            self.bot = None
            print(f"Error loading backend: {e}")

    def setup_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#1E3A8A", height=70)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="🤖 Live NLP AI Chatbot",
            font=("Segoe UI", 16, "bold"),
            bg="#1E3A8A",
            fg="#FFFFFF"
        )
        title_lbl.pack(pady=(12, 2))

        subtitle_lbl = tk.Label(
            header,
            text="Powered by NLTK, TF-IDF Vectorizer & Cosine Similarity",
            font=("Segoe UI", 9),
            bg="#1E3A8A",
            fg="#93C5FD"
        )
        subtitle_lbl.pack(pady=(0, 10))

        # Chat Conversation Area
        chat_frame = tk.Frame(self.root, bg="#F3F4F6")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.chat_history = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg="#FFFFFF",
            fg="#1F2937",
            relief=tk.FLAT,
            padx=12,
            pady=12
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        self.chat_history.config(state=tk.DISABLED)

        # Configure Text Tags for Styling
        self.chat_history.tag_config("user_tag", foreground="#1D4ED8", font=("Segoe UI", 10, "bold"))
        self.chat_history.tag_config("bot_tag", foreground="#059669", font=("Segoe UI", 10, "bold"))
        self.chat_history.tag_config("msg_user", foreground="#111827", spacing3=6)
        self.chat_history.tag_config("msg_bot", foreground="#1F2937", spacing3=8)
        self.chat_history.tag_config("system_tag", foreground="#6B7280", font=("Segoe UI", 9, "italic"), spacing3=8)

        # Initial Welcome Message
        self.display_message("System", "Chatbot initialized! Ask any question about AI, NLP, Chatbots, or Machine Learning.\n", "system_tag")

        # Bottom Input Area
        input_frame = tk.Frame(self.root, bg="#F3F4F6")
        input_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        self.user_entry = tk.Entry(
            input_frame,
            font=("Segoe UI", 11),
            bg="#FFFFFF",
            fg="#111827",
            relief=tk.SOLID,
            bd=1
        )
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
        self.user_entry.bind("<Return>", self.send_message)
        self.user_entry.focus()

        send_btn = tk.Button(
            input_frame,
            text="Send",
            font=("Segoe UI", 10, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=18,
            cursor="hand2",
            command=self.send_message
        )
        send_btn.pack(side=tk.RIGHT, ipady=4)

    def display_message(self, sender, text, tag):
        self.chat_history.config(state=tk.NORMAL)
        if sender == "You":
            self.chat_history.insert(tk.END, "You: ", "user_tag")
            self.chat_history.insert(tk.END, f"{text}\n", "msg_user")
        elif sender == "Bot":
            self.chat_history.insert(tk.END, "Bot: ", "bot_tag")
            self.chat_history.insert(tk.END, f"{text}\n\n", "msg_bot")
        else:
            self.chat_history.insert(tk.END, f"{text}\n", "system_tag")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.see(tk.END)

    def send_message(self, event=None):
        user_text = self.user_entry.get().strip()
        if not user_text:
            return

        self.user_entry.delete(0, tk.END)
        self.display_message("You", user_text, "user_tag")

        if user_text.lower() in ["bye", "exit", "quit", "goodbye"]:
            self.display_message("Bot", "Goodbye! Have a great day ahead!", "bot_tag")
            return

        if not self.bot:
            self.display_message("Bot", "Backend not ready. Please ensure corpus.txt exists.", "bot_tag")
            return

        # Run response generation in background thread to keep GUI responsive
        threading.Thread(target=self._process_response, args=(user_text,), daemon=True).start()

    def _process_response(self, text):
        response = self.bot.generate_response(text)
        self.root.after(0, lambda: self.display_message("Bot", response, "bot_tag"))

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
