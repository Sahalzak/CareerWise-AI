import sqlite3
from datetime import datetime

conn = sqlite3.connect("careerwise.db")
c = conn.cursor()

# Check if column exists first
c.execute("PRAGMA table_info(users)")
columns = [col[1] for col in c.fetchall()]

if "created_at" not in columns:
    # Step 1: Add column (no default value)
    c.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
    print("✅ Added 'created_at' column to users table.")

    # Step 2: Fill in existing rows with current datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE users SET created_at = ?", (now,))
    print(f"🕒 Set created_at for existing users as {now}.")
else:
    print("✅ Column 'created_at' already exists.")

conn.commit()
conn.close()
