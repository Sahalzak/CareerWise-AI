import tkinter as tk
import pyttsx3

def speak_text():
    text = entry.get()
    if text:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

root = tk.Tk()
root.title("Text to Speech Converter")
root.geometry("350x200")

tk.Label(root, text="Enter Text:").pack(pady=10)
entry = tk.Entry(root, width=40)
entry.pack(pady=5)

tk.Button(root, text="Speak", command=speak_text, bg="lightblue").pack(pady=10)
root.mainloop()