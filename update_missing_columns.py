import sqlite3

conn = sqlite3.connect("careerwise.db")
c = conn.cursor()

# Add any missing columns
columns_to_add = [
    ("work_env", "TEXT"),
    ("relocation", "TEXT"),
    ("internship_interest", "TEXT"),
    ("extra_curricular", "TEXT"),
    ("personality", "TEXT"),
    ("preferred_industry", "TEXT")
]

for col, col_type in columns_to_add:
    try:
        c.execute(f"ALTER TABLE profiles ADD COLUMN {col} {col_type}")
        print(f"✅ Added column: {col}")
    except sqlite3.OperationalError:
        print(f"⚠️ Column already exists: {col}")

conn.commit()
conn.close()
print("✅ All columns verified or added successfully.")
