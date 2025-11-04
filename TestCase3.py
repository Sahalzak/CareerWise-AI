import tkinter as tk

def calculate_bill():
    try:
        units = int(entry_units.get())
        if units <= 100:
            amount = units * 5
        else:
            amount = (100 * 5) + (units - 100) * 7
        label_result.config(text=f"Total Bill: ₹{amount}")
    except ValueError:
        label_result.config(text="Enter valid units!")

root = tk.Tk()
root.title("Electricity Bill Estimator")
root.geometry("350x250")

tk.Label(root, text="Enter Units Consumed:").pack(pady=10)
entry_units = tk.Entry(root)
entry_units.pack(pady=5)

tk.Button(root, text="Calculate Bill", command=calculate_bill, bg="orange").pack(pady=10)
label_result = tk.Label(root, text="")
label_result.pack(pady=5)

root.mainloop()