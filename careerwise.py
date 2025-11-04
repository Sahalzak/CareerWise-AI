"""
CareerWise AI - Streamlit prototype (single-file)
Features:
 - SQLite user registration/login (hashed passwords)
 - Profile collection (education, skills, interests)
 - Synthetic dataset generator and ML model training (RandomForest)
 - Career prediction with confidence scores (top-3)
 - Save feedback and admin dashboard (view users/feedback/retrain)
Note: For production, replace synthetic data with real labeled data,
use proper authentication, HTTPS, and stronger security practices.
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import hashlib
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from io import BytesIO

# ---------------------------
# Constants / Paths
# ---------------------------
DB_PATH = "careerwise.db"
MODEL_PATH = "careerwise_model.joblib"

# ---------------------------
# Database helpers
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # users table: id, name, email (unique), password_hash
    c.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT
        )
    ''')
    # profiles table: user_id foreign key, age, education, skills (csv), interests (csv), gpa
    c.execute('''
        CREATE TABLE IF NOT EXISTS profiles(
            user_id INTEGER PRIMARY KEY,
            age INTEGER,
            education TEXT,
            gpa REAL,
            skills TEXT,
            interests TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    # feedback table
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            career_suggestion TEXT,
            rating INTEGER,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users(name,email,password_hash) VALUES (?,?,?)",
                  (name, email, hash_password(password)))
        conn.commit()
        user_id = c.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return None  # email exists
    conn.close()
    return user_id

def authenticate_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, password_hash FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    if row and hash_password(password) == row[2]:
        return {"id": row[0], "name": row[1], "email": email}
    return None

def save_profile(user_id, age, education, gpa, skills, interests):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # upsert
    c.execute('''
        INSERT INTO profiles(user_id,age,education,gpa,skills,interests)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
        age=excluded.age,education=excluded.education,gpa=excluded.gpa,skills=excluded.skills,interests=excluded.interests
    ''', (user_id, age, education, gpa, ",".join(skills), ",".join(interests)))
    conn.commit()
    conn.close()

def get_profile(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT age,education,gpa,skills,interests FROM profiles WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"age": row[0], "education": row[1], "gpa": row[2], "skills": (row[3] or "").split(","), "interests": (row[4] or "").split(",")}
    return None

def save_feedback(user_id, career_suggestion, rating, comment):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO feedback(user_id,career_suggestion,rating,comment) VALUES (?,?,?,?)",
              (user_id, career_suggestion, rating, comment))
    conn.commit()
    conn.close()

def fetch_users():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id,name,email FROM users", conn)
    conn.close()
    return df

def fetch_feedback():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id,user_id,career_suggestion,rating,comment,timestamp FROM feedback ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# ---------------------------
# Synthetic dataset & model
# ---------------------------
CAREER_LABELS = [
    "Software Developer",
    "Data Analyst",
    "AI/ML Engineer",
    "UX Designer",
    "Cybersecurity Specialist",
    "Network Engineer",
    "Product Manager",
    "Quality Assurance",
    "Systems Administrator",
    "Business Analyst"
]

SKILL_POOL = ["python","java","c++","sql","ml","dl","statistics","excel","html","css","js","ux","cybersec","networks","cloud","testing","pm"]

def generate_synthetic_dataset(n_samples=2000, random_state=42):
    """
    Create a synthetic labeled dataset that maps simple features to career labels.
    Features:
     - education: ['High School','Diploma','B.Tech','M.Tech','BSc','MSc']
     - gpa: 0-10
     - skills: one-hot for small set of grouped skills
     - interests: domain interests
    """
    rng = np.random.RandomState(random_state)
    edu_choices = ['High School','Diploma','B.Tech','M.Tech','BSc','MSc']
    interest_choices = ['Programming','Design','Analytics','Security','Networking','Management','Testing']

    rows = []
    for i in range(n_samples):
        edu = rng.choice(edu_choices, p=[0.05,0.1,0.45,0.05,0.2,0.15])
        gpa = round(rng.normal(7.0, 1.0), 2)
        # choose some skills
        num_skills = rng.randint(1,5)
        skills = list(rng.choice(SKILL_POOL, size=num_skills, replace=False))
        interest = rng.choice(interest_choices)
        # heuristic label mapping (very simplified)
        if 'ml' in skills or 'dl' in skills or (interest=='Analytics' and 'python' in skills):
            label = "AI/ML Engineer" if rng.rand() < 0.6 else "Data Analyst"
        elif 'ux' in skills or interest=='Design':
            label = "UX Designer"
        elif 'cybersec' in skills or interest=='Security':
            label = "Cybersecurity Specialist"
        elif 'sql' in skills or 'excel' in skills or interest=='Analytics':
            label = "Data Analyst"
        elif 'html' in skills or 'css' in skills or 'js' in skills:
            label = "Software Developer"
        elif 'testing' in skills:
            label = "Quality Assurance"
        elif interest=='Management' or 'pm' in skills:
            label = "Product Manager"
        elif 'networks' in skills or interest=='Networking':
            label = "Network Engineer"
        elif 'cloud' in skills:
            label = "Systems Administrator"
        else:
            label = rng.choice(CAREER_LABELS)
        rows.append({
            "education": edu,
            "gpa": max(0.0, min(10.0, gpa)),
            "skills": "|".join(skills),
            "interest": interest,
            "label": label
        })
    df = pd.DataFrame(rows)
    return df

def build_feature_dataframe(df):
    """ Turn the synthetic dataset into numeric features for model training """
    # Expand skills into multi-hot for SKILL_POOL subset
    def skill_vector(skills_str):
        s = set(skills_str.split("|"))
        return [1 if sk in s else 0 for sk in SKILL_POOL]
    skills_mat = np.array([skill_vector(s) for s in df['skills']])
    # education one-hot
    edu_ohe = pd.get_dummies(df['education'], prefix='edu')
    # interest one-hot
    int_ohe = pd.get_dummies(df['interest'], prefix='int')
    # numeric gpa
    gpa = df[['gpa']].values
    X = np.hstack([gpa, skills_mat, edu_ohe.values, int_ohe.values])
    feature_names = ['gpa'] + [f"skill_{s}" for s in SKILL_POOL] + list(edu_ohe.columns) + list(int_ohe.columns)
    return pd.DataFrame(X, columns=feature_names), df['label']

def train_and_save_model(df=None):
    if df is None:
        df = generate_synthetic_dataset()
    X, y = build_feature_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=1)
    clf.fit(X_train, y_train)
    # simple evaluation
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, zero_division=0)
    # Save pipeline: we'll store feature columns + classifier
    model_package = {
        "model": clf,
        "feature_cols": list(X.columns)
    }
    joblib.dump(model_package, MODEL_PATH)
    return report

def load_model():
    if not os.path.exists(MODEL_PATH):
        # train default if missing
        train_and_save_model()
    package = joblib.load(MODEL_PATH)
    return package["model"], package["feature_cols"]

def profile_to_feature_vector(profile, feature_cols):
    """
    Convert a saved user profile (dict) to model feature vector consistent with model training features.
    profile: {age,education,gpa,skills(list),interests(list)}
    """
    # gpa
    gpa_val = float(profile.get("gpa", 0.0))
    # skills multi-hot
    skill_set = set([s.lower() for s in profile.get("skills", [])])
    skills_vec = [1 if sk in skill_set else 0 for sk in SKILL_POOL]
    # education one-hot columns present in feature_cols
    edu_cols = [c for c in feature_cols if c.startswith("edu_")]
    int_cols = [c for c in feature_cols if c.startswith("int_")]
    edu_map = {c:0 for c in edu_cols}
    int_map = {c:0 for c in int_cols}
    user_edu = profile.get("education","")
    user_int = profile.get("interests",[""])[0] if isinstance(profile.get("interests",[]), list) else profile.get("interests","")
    col_edu = f"edu_{user_edu}"
    col_int = f"int_{user_int}"
    if col_edu in edu_map: edu_map[col_edu] = 1
    if col_int in int_map: int_map[col_int] = 1
    vec = [gpa_val] + skills_vec + [edu_map[c] for c in edu_cols] + [int_map[c] for c in int_cols]
    # align with feature_cols order (they should match)
    return np.array(vec).reshape(1,-1)

def predict_careers(profile, top_k=3):
    model, feature_cols = load_model()
    x = profile_to_feature_vector(profile, feature_cols)
    probs = model.predict_proba(x)[0]
    classes = model.classes_
    # pair and sort
    paired = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return paired[:top_k], dict(zip(classes, probs))

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="CareerWise AI (Prototype)", layout="centered")

init_db()  # ensure DB exists

st.title("CareerWise AI — Prototype")
st.caption("A demo single-file implementation (Streamlit). Not for production use.")

# Sidebar navigation
menu = st.sidebar.selectbox("Navigation", ["Home", "Register", "Login", "Admin (Secure)"])

if menu == "Home":
    st.header("Welcome")
    st.write("""
    CareerWise AI helps generate career suggestions based on a simple profile.
    Try Register -> Login -> Fill Profile -> Get Recommendations.
    For demo purposes, model is trained on a synthetic dataset.
    """)

elif menu == "Register":
    st.header("Create an account")
    with st.form("register_form"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register")
    if submitted:
        if not (name and email and password):
            st.error("Please fill all fields.")
        else:
            new_id = register_user(name, email, password)
            if new_id is None:
                st.error("Email already registered. Try logging in.")
            else:
                st.success("Registered successfully! You may now login.")

elif menu == "Login":
    st.header("Login to your account")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        user = authenticate_user(email, password)
        if user:
            st.success(f"Welcome, {user['name']}!")
            # set session state
            st.session_state['user'] = user
        else:
            st.error("Invalid credentials.")

    # if logged in in this session
    if 'user' in st.session_state:
        user = st.session_state['user']
        st.subheader("Your Dashboard")
        st.write(f"Logged in as: **{user['name']}** ({user['email']})")
        profile = get_profile(user['id'])
        if profile:
            st.info("Profile found. You can update it below.")
        else:
            st.info("No profile found. Please fill your details.")

        with st.form("profile_form"):
            age = st.number_input("Age", min_value=12, max_value=100, value=profile['age'] if profile else 18)
            education = st.selectbox("Highest Education", options=['High School','Diploma','B.Tech','M.Tech','BSc','MSc'], index=2)
            gpa = st.number_input("GPA / Percentage (0-10)", min_value=0.0, max_value=10.0, value=float(profile['gpa']) if profile and profile['gpa'] else 7.0, step=0.1)
            # skills multiselect
            skills = st.multiselect("Select your skills (choose all that apply)", options=SKILL_POOL, default=profile['skills'] if profile and profile['skills'] and profile['skills'][0] else [])
            interests = st.multiselect("Interests / Preferred domain", options=['Programming','Design','Analytics','Security','Networking','Management','Testing'], default=profile['interests'] if profile and profile['interests'] and profile['interests'][0] else [])
            saved = st.form_submit_button("Save Profile")
        if saved:
            save_profile(user['id'], int(age), education, float(gpa), skills, interests)
            st.success("Profile saved.")

        # Option to train/retrain model locally (user-triggered)
        st.markdown("---")
        st.subheader("Career Suggestion")
        if st.button("Get Career Recommendations"):
            prof = get_profile(user['id'])
            if not prof:
                st.warning("Please save your profile first.")
            else:
                # format profile
                prepared_profile = {
                    "gpa": prof['gpa'] or 0.0,
                    "education": prof['education'] or 'B.Tech',
                    "skills": prof['skills'] if prof['skills'] else [],
                    "interests": prof['interests'] if prof['interests'] else []
                }
                try:
                    top_k, probs_map = predict_careers(prepared_profile, top_k=5)
                except Exception as e:
                    st.error("Model not found or needs training. Training now...")
                    report = train_and_save_model()
                    st.text("Training report (initial):")
                    st.text(report)
                    top_k, probs_map = predict_careers(prepared_profile, top_k=5)

                st.write("### Recommended Careers (Top results)")
                for career, score in top_k:
                    st.markdown(f"**{career}** — Confidence: **{score*100:.1f}%**")
                    # Add short boilerplate descriptions
                    if career == "Software Developer":
                        st.caption("Typical skills: programming (Python/Java/C++), data structures, web development.")
                    elif career == "Data Analyst":
                        st.caption("Typical skills: SQL, Excel, data visualization, statistics.")
                    elif career == "AI/ML Engineer":
                        st.caption("Typical skills: python, ML frameworks, statistics, deep learning.")
                    elif career == "UX Designer":
                        st.caption("Typical skills: user research, prototyping, Figma, design thinking.")
                    elif career == "Cybersecurity Specialist":
                        st.caption("Typical skills: network security, penetration testing, security tools.")
                # provide download of recommendations as CSV
                rec_df = pd.DataFrame(top_k, columns=["Career","Confidence"])
                buf = BytesIO()
                rec_df.to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download Recommendations (CSV)", data=buf, file_name="career_recommendations.csv", mime="text/csv")

                # Save last prediction in session for feedback
                st.session_state['last_recommendation'] = rec_df

        # Feedback form
        st.markdown("----")
        st.subheader("Give Feedback on Recommendation")
        if 'last_recommendation' in st.session_state:
            last = st.session_state['last_recommendation']
            career_choice = st.selectbox("Which recommended career would you like to rate?", options=list(last['Career']))
            rating = st.slider("Rate usefulness (1 - Not useful, 5 - Very useful)", min_value=1, max_value=5, value=4)
            comment = st.text_area("Optional comment")
            if st.button("Submit Feedback"):
                save_feedback(user['id'], career_choice, int(rating), comment)
                st.success("Thanks for your feedback! It will help improve recommendations.")
        else:
            st.info("Get recommendations first to provide feedback.")

elif menu == "Admin (Secure)":
    st.header("Admin / System Management")
    st.write("Admin access: enter admin secret key (demo only).")
    secret = st.text_input("Admin Secret Key", type="password")
    # Demo secret - in real app use secure auth
    if secret == "admin123":
        st.success("Admin authenticated (demo).")
        st.subheader("Users")
        users_df = fetch_users()
        st.dataframe(users_df)
        st.subheader("Feedback")
        fb = fetch_feedback()
        st.dataframe(fb)
        st.subheader("Train / Retrain Model")
        st.write("You may retrain the model on a freshly generated synthetic dataset (demo). In production, retrain on clean labelled data.")
        if st.button("Train model (generate synthetic data -> train)"):
            with st.spinner("Training model..."):
                df = generate_synthetic_dataset(n_samples=3000)
                report = train_and_save_model(df)
                st.success("Model trained and saved.")
                st.text("Evaluation report on test split:")
                st.text(report)
        st.subheader("Export Feedback as CSV")
        if st.button("Download Feedback CSV"):
            data = fetch_feedback()
            st.download_button("Download feedback", data=data.to_csv(index=False).encode('utf-8'), file_name="feedback.csv", mime="text/csv")
    else:
        if secret:
            st.error("Invalid admin key.")

# Footer / Helpful notes
st.markdown("---")
st.caption("Prototype by CareerWise AI — Streamlit demo. Replace synthetic data with real labeled datasets for production. Secure credentials and deploy with HTTPS.")
