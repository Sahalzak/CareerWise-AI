import tkinter as tk
from tkinter import messagebox

def calculate_discount():
    try:
        original_price = float(entry_price.get())
        discount_percent = float(entry_discount.get())
        
        discount_amount = (original_price * discount_percent) / 100
        final_price = original_price - discount_amount

        label_result.config(
            text=f"Discount Amount: ₹{discount_amount:.2f}\nFinal Price: ₹{final_price:.2f}"
        )
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers.")

# Create main window
root = tk.Tk()
root.title("Discount Price Calculator")
root.geometry("350x250")
root.config(bg="#f2f2f2")

# Heading
tk.Label(root, text="Discount Price Calculator", font=("Arial", 14, "bold"), bg="#f2f2f2", fg="#333").pack(pady=10)

# Original Price
tk.Label(root, text="Original Price (₹):", bg="#f2f2f2").pack()
entry_price = tk.Entry(root, width=25)
entry_price.pack(pady=5)

# Discount Percentage
tk.Label(root, text="Discount Percentage (%):", bg="#f2f2f2").pack()
entry_discount = tk.Entry(root, width=25)
entry_discount.pack(pady=5)

# Button
tk.Button(root, text="Calculate Discount", command=calculate_discount, bg="#4CAF50", fg="white", width=20).pack(pady=10)

# Result Label
label_result = tk.Label(root, text="", bg="#f2f2f2", font=("Arial", 11))
label_result.pack(pady=10)

# Run window
root.mainloop()
