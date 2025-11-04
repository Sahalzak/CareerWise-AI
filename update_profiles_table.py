import sqlite3

conn = sqlite3.connect("careerwise.db")
c = conn.cursor()

# Check existing columns
c.execute("PRAGMA table_info(profiles)")
columns = [col[1] for col in c.fetchall()]
print("Existing columns:", columns)

# Add missing column if needed
if "coding_confidence" not in columns:
    print("Adding 'coding_confidence' column...")
    c.execute("ALTER TABLE profiles ADD COLUMN coding_confidence TEXT;")
    conn.commit()
    print("✅ Column added successfully.")
else:
    print("✅ Column already exists.")

conn.close()
