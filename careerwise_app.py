# careerwise_app.py
"""
CareerWise AI — complete working Streamlit single-file app
- Register / Login / Profile / Recommend / Feedback / Admin
- SQLite DB (careerwise.db) with automatic init
- Synthetic ML model (RandomForest) for demo predictions
Note: Demo only — do not use as production auth/security.
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import hashlib
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from io import BytesIO
import os
st.write("Using database path:", os.path.abspath("careerwise.db"))

# ---------------------------
# Configuration & constants
# ---------------------------
st.set_page_config(page_title="CareerWise AI", layout="wide", initial_sidebar_state="expanded")
DB_PATH = "careerwise.db"
MODEL_PATH = "careerwise_model.joblib"
ADMIN_KEY = os.getenv("CAREERWISE_ADMIN_KEY", "admin123")  # for demo

CAREER_LABELS = [
    "Software Developer", "Data Analyst", "AI/ML Engineer", "UX Designer",
    "Cybersecurity Specialist", "Network Engineer", "Product Manager",
    "Quality Assurance", "Systems Administrator", "Business Analyst"
]

SKILL_POOL = [
    "python","java","c++","sql","ml","dl","statistics","excel","html","css","js",
    "ux","cybersec","networks","cloud","testing","pm","docker","git","linux"
]

INTEREST_OPTIONS = [
    'Programming','Design','Analytics','Security','Networking',
    'Management','Testing','Cloud','DevOps','Research'
]

# ---------------------------
# Database initialization
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # profiles table
    c.execute('''
        CREATE TABLE IF NOT EXISTS profiles(
            user_id INTEGER PRIMARY KEY,
            age INTEGER,
            education TEXT,
            gpa REAL,
            skills TEXT,
            interests TEXT,
            coding_confidence INTEGER,
            work_env TEXT,
            relocation INTEGER,
            internship_interest INTEGER,
            extra_curricular TEXT,
            personality TEXT,
            preferred_industry TEXT,
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

# ensure DB ready
init_db()

# ---------------------------
# DB helper functions
# ---------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users(name,email,password_hash) VALUES (?,?,?)",
                  (name, email, hash_password(password)))
        conn.commit()
        uid = c.lastrowid
        conn.close()
        return uid
    except sqlite3.IntegrityError:
        return None

def authenticate_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, password_hash FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    if row and hash_password(password) == row[2]:
        return {"id": row[0], "name": row[1], "email": email}
    return None

def save_profile(user_id, profile_dict):
    # upsert into profiles
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO profiles(user_id, age, education, gpa, skills, interests, coding_confidence,
            work_env, relocation, internship_interest, extra_curricular, personality, preferred_industry)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            age=excluded.age,
            education=excluded.education,
            gpa=excluded.gpa,
            skills=excluded.skills,
            interests=excluded.interests,
            coding_confidence=excluded.coding_confidence,
            work_env=excluded.work_env,
            relocation=excluded.relocation,
            internship_interest=excluded.internship_interest,
            extra_curricular=excluded.extra_curricular,
            personality=excluded.personality,
            preferred_industry=excluded.preferred_industry
    ''', (
        user_id,
        profile_dict.get('age'),
        profile_dict.get('education'),
        profile_dict.get('gpa'),
        ",".join(profile_dict.get('skills', [])),
        ",".join(profile_dict.get('interests', [])),
        profile_dict.get('coding_confidence'),
        profile_dict.get('work_env'),
        int(profile_dict.get('relocation', 0)),
        int(profile_dict.get('internship_interest', 0)),
        profile_dict.get('extra_curricular', ''),
        ",".join([f"{k}:{v}" for k,v in profile_dict.get('personality', {}).items()]),
        profile_dict.get('preferred_industry', '')
    ))
    conn.commit()
    conn.close()

def get_profile(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT age, education, gpa, skills, interests, coding_confidence, work_env, relocation,
               internship_interest, extra_curricular, personality, preferred_industry
        FROM profiles WHERE user_id=?
    """, (user_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    profile = {
        "age": row[0],
        "education": row[1],
        "gpa": row[2],
        "skills": (row[3] or "").split(",") if row[3] else [],
        "interests": (row[4] or "").split(",") if row[4] else [],
        "coding_confidence": row[5],
        "work_env": row[6],
        "relocation": bool(row[7]),
        "internship_interest": bool(row[8]),
        "extra_curricular": row[9] or "",
        "personality": {},
        "preferred_industry": row[11] or ""
    }
    # parse personality string if present (format: key:val,key:val)
    if row[10]:
        try:
            personality = {}
            parts = row[10].split(",")
            for p in parts:
                if ":" in p:
                    k,v = p.split(":")
                    personality[k] = float(v)
            profile["personality"] = personality
        except Exception:
            profile["personality"] = {}
    return profile

def save_feedback(user_id, career_suggestion, rating, comment):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO feedback(user_id, career_suggestion, rating, comment) VALUES (?,?,?,?)",
              (user_id, career_suggestion, rating, comment))
    conn.commit()
    conn.close()

def fetch_users():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, name, email, created_at FROM users ORDER BY id DESC", conn)
    conn.close()
    return df

def fetch_feedback():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, user_id, career_suggestion, rating, comment, timestamp FROM feedback ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# ---------------------------
# Synthetic dataset & model
# ---------------------------
def generate_synthetic_dataset(n_samples=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    edu_choices = ['High School','Diploma','B.Tech','M.Tech','BSc','MSc']
    rows = []
    for _ in range(n_samples):
        edu = rng.choice(edu_choices, p=[0.05,0.1,0.45,0.05,0.2,0.15])
        gpa = float(np.clip(rng.normal(7.0, 1.1), 0.0, 10.0))
        num_skills = rng.randint(1,6)
        skills = list(rng.choice(SKILL_POOL, size=num_skills, replace=False))
        interest = rng.choice(INTEREST_OPTIONS)
        coding_conf = int(np.clip(rng.normal(6.0,2.0), 1, 10))
        relocation = int(rng.rand() > 0.6)
        internship = int(rng.rand() > 0.5)
        # heuristic label
        sset = set(skills)
        if 'ml' in sset or 'dl' in sset:
            label = rng.choice(["AI/ML Engineer","Data Analyst"], p=[0.65,0.35])
        elif 'ux' in sset or interest == 'Design':
            label = "UX Designer"
        elif 'cybersec' in sset or interest == 'Security':
            label = "Cybersecurity Specialist"
        elif 'sql' in sset or 'excel' in sset or interest == 'Analytics':
            label = "Data Analyst"
        elif 'html' in sset or 'css' in sset or 'js' in sset:
            label = "Software Developer"
        else:
            label = rng.choice(CAREER_LABELS)
        rows.append({
            "education": edu,
            "gpa": gpa,
            "skills": "|".join(skills),
            "interest": interest,
            "coding_confidence": coding_conf,
            "relocation": relocation,
            "internship": internship,
            "label": label
        })
    return pd.DataFrame(rows)

def build_feature_dataframe(df):
    # create feature matrix (simple)
    def skill_vector(skills_str):
        s = set(skills_str.split("|"))
        return [1 if sk in s else 0 for sk in SKILL_POOL]
    skills_mat = np.array([skill_vector(s) for s in df['skills']])
    edu_ohe = pd.get_dummies(df['education'], prefix='edu')
    int_ohe = pd.get_dummies(df['interest'], prefix='int')
    gpa = df[['gpa']].values
    coding = df[['coding_confidence']].values
    relocation = df[['relocation']].values
    internship = df[['internship']].values
    X = np.hstack([gpa, coding, relocation, internship, skills_mat, edu_ohe.values, int_ohe.values])
    feature_names = ['gpa','coding_confidence','relocation','internship'] + [f"skill_{s}" for s in SKILL_POOL] + list(edu_ohe.columns) + list(int_ohe.columns)
    return pd.DataFrame(X, columns=feature_names), df['label']

def train_and_save_model(df=None):
    if df is None:
        df = generate_synthetic_dataset(n_samples=2000)
    X, y = build_feature_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=1, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, zero_division=0)
    joblib.dump({"model": clf, "feature_cols": list(X.columns)}, MODEL_PATH)
    return report

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    pkg = joblib.load(MODEL_PATH)
    return pkg["model"], pkg["feature_cols"]

def profile_to_feature_vector(profile, feature_cols):
    gpa_val = float(profile.get("gpa", 0.0))
    coding = int(profile.get("coding_confidence", 5))
    relocation = int(profile.get("relocation", False))
    internship = int(profile.get("internship_interest", False))
    skill_set = set([s.lower() for s in profile.get("skills", [])])
    skills_vec = [1 if sk in skill_set else 0 for sk in SKILL_POOL]
    edu_cols = [c for c in feature_cols if c.startswith("edu_")]
    int_cols = [c for c in feature_cols if c.startswith("int_")]
    edu_map = {c:0 for c in edu_cols}
    int_map = {c:0 for c in int_cols}
    user_edu = profile.get("education","")
    user_int = profile.get("interests",[""])[0] if isinstance(profile.get("interests",[]), list) and profile.get("interests") else profile.get("interests","")
    col_edu = f"edu_{user_edu}"
    col_int = f"int_{user_int}"
    if col_edu in edu_map: edu_map[col_edu] = 1
    if col_int in int_map: int_map[col_int] = 1
    vec = [gpa_val, coding, relocation, internship] + skills_vec + [edu_map[c] for c in edu_cols] + [int_map[c] for c in int_cols]
    return np.array(vec).reshape(1,-1)

def predict_careers(profile, top_k=3):
    model, feature_cols = load_model()
    x = profile_to_feature_vector(profile, feature_cols)
    probs = model.predict_proba(x)[0]
    classes = model.classes_
    paired = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return paired[:top_k], dict(zip(classes, probs))

# ---------------------------
# UI helpers
# ---------------------------
def get_boilerplate(career_name):
    mapping = {
        "Software Developer": "Core skills: programming (Python/Java/C++), data structures, algorithms, web development.",
        "Data Analyst": "Core skills: SQL, Excel, data visualization, statistics, business understanding.",
        "AI/ML Engineer": "Core skills: python, ML frameworks (scikit-learn/TensorFlow), math/statistics.",
        "UX Designer": "Core skills: user research, prototyping, interaction design, Figma.",
        "Cybersecurity Specialist": "Core skills: network security, penetration testing, incident response.",
        "Network Engineer": "Core skills: network design, routing/switching, protocols, troubleshooting.",
        "Product Manager": "Core skills: communication, stakeholder management, product design, analytics.",
        "Quality Assurance": "Core skills: testing methodologies, automation basics, attention to detail.",
        "Systems Administrator": "Core skills: Linux, cloud basics, scripting, system monitoring.",
        "Business Analyst": "Core skills: requirement gathering, stakeholder analysis, data-driven decisions."
    }
    return mapping.get(career_name, "")

def inject_css():
    st.markdown("""
        <style>
        .card { padding:16px; border-radius:8px; background:white; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
        .small-muted { color:#6b7280; font-size:0.9rem; }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------
# App layout & pages
# ---------------------------
inject_css()
st.title("CareerWise AI — Prototype")
st.caption("Smart, data-driven career suggestions (demo).")

menu = st.sidebar.selectbox("Navigation", ["Home", "Register", "Login", "Admin"])

if menu == "Home":
    st.header("Welcome")
    st.write("CareerWise AI provides career suggestions based on your profile. Register → Login → Fill profile → Get recommendations.")
    st.markdown("---")
    st.subheader("How it works")
    st.write("- Fill profile (education, skills, interests, preferences).")
    st.write("- The model uses a synthetic dataset for demo predictions.")
    st.write("- Provide feedback to help improve results.")

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
            st.session_state['user'] = user
        else:
            st.error("Invalid credentials.")

    # if logged in
    if 'user' in st.session_state:
        user = st.session_state['user']
        st.subheader("Your Dashboard")
        st.write(f"Logged in as: **{user['name']}** ({user['email']})")
        profile = get_profile(user['id'])
        if profile:
            st.info("Profile found. You can update it below.")
        else:
            st.info("No profile found. Please fill your details.")

        # profile form (two-column)
        col_left, col_right = st.columns([2,1])
        with col_left:
            with st.form("profile_form"):
                st.markdown("**Personal & Academic**")
                age = st.number_input("Age", min_value=12, max_value=100, value=profile['age'] if profile and profile.get('age') else 18)
                education = st.selectbox("Highest Education", options=['High School','Diploma','B.Tech','M.Tech','BSc','MSc'], index=2)
                gpa = st.number_input("GPA / Percentage (0-10)", min_value=0.0, max_value=10.0,
                                      value=float(profile['gpa']) if profile and profile.get('gpa') else 7.0, step=0.1)
                st.markdown("**Skills & Interests**")
                skills = st.multiselect("Select your skills (choose all that apply)", options=SKILL_POOL,
                                        default=profile['skills'] if profile else [])
                interests = st.multiselect("Interests / Preferred domain", options=INTEREST_OPTIONS,
                                           default=profile['interests'] if profile else [])
                st.markdown("**Preferences & Personality**")
                coding_conf = st.slider("Coding confidence (1 = beginner, 10 = expert)", 1, 10,
                                        value=int(profile['coding_confidence']) if profile and profile.get('coding_confidence') else 6)
                work_env = st.selectbox("Preferred work environment", ["Startup/Agile","Corporate","Research/Academia","Remote/Freelance","Hybrid"], index=0)
                relocation = st.selectbox("Willing to relocate?", ["No","Yes"], index=1 if profile and profile.get('relocation') else 0)
                internship = st.selectbox("Open to internships / part-time roles?", ["No","Yes"], index=1 if profile and profile.get('internship_interest') else 0)
                preferred_industry = st.selectbox("Preferred industry (optional)", ["Any","Tech","Finance","Healthcare","Education","E-commerce","Telecom","Government"], index=0)
                extra_curricular = st.text_area("Extra-curricular / Certifications (comma-separated)", value=profile['extra_curricular'] if profile else "")
                with st.expander("Personality quick survey (optional)"):
                    p_extrav = st.slider("Social / Extraversion", 0.0, 1.0, value=profile.get('personality', {}).get('extrav',0.5) if profile else 0.5, step=0.1)
                    p_consc = st.slider("Organized / Conscientious", 0.0, 1.0, value=profile.get('personality', {}).get('consc',0.6) if profile else 0.6, step=0.1)
                    p_open = st.slider("Curiosity / Openness", 0.0, 1.0, value=profile.get('personality', {}).get('open',0.6) if profile else 0.6, step=0.1)
                    p_agree = st.slider("Cooperative / Agreeableness", 0.0, 1.0, value=profile.get('personality', {}).get('agree',0.5) if profile else 0.5, step=0.1)
                    p_neuro = st.slider("Stress-resilience (lower is sensitive)", 0.0, 1.0, value=profile.get('personality', {}).get('neuro',0.4) if profile else 0.4, step=0.1)
                personality = {'extrav': p_extrav, 'consc': p_consc, 'open': p_open, 'agree': p_agree, 'neuro': p_neuro}
                save = st.form_submit_button("Save Profile")
            if save:
                profile_dict = {
                    "age": int(age), "education": education, "gpa": float(gpa),
                    "skills": skills, "interests": interests, "coding_confidence": int(coding_conf),
                    "work_env": work_env, "relocation": True if relocation=="Yes" else False,
                    "internship_interest": True if internship=="Yes" else False,
                    "extra_curricular": extra_curricular, "personality": personality,
                    "preferred_industry": preferred_industry
                }
                save_profile(user['id'], profile_dict)
                st.success("Profile saved.")

        with col_right:
            st.markdown("**Quick actions**")
            if st.button("Get Career Recommendations"):
                prof = get_profile(user['id'])
                if not prof:
                    st.warning("Please save your profile first.")
                else:
                    p = st.progress(0)
                    for i in range(4):
                        p.progress((i+1)/4)
                    prepared_profile = {
                        "gpa": prof.get('gpa', 0.0),
                        "education": prof.get('education', 'B.Tech'),
                        "skills": prof.get('skills', []),
                        "interests": prof.get('interests', []),
                        "coding_confidence": prof.get('coding_confidence', 5),
                        "relocation": prof.get('relocation', False),
                        "internship_interest": prof.get('internship_interest', False)
                    }
                    try:
                        top_k, probs_map = predict_careers(prepared_profile, top_k=5)
                    except Exception:
                        st.warning("Model not found. Training model now (demo)...")
                        report = train_and_save_model()
                        st.text(report)
                        top_k, probs_map = predict_careers(prepared_profile, top_k=5)

                    st.markdown("### Recommended Careers (Top results)")
                    for career, score in top_k:
                        st.markdown(f"**{career}** — Confidence: **{score*100:.1f}%**")
                        st.caption(get_boilerplate(career))
                    rec_df = pd.DataFrame(top_k, columns=["Career","Confidence"])
                    buf = BytesIO()
                    rec_df.to_csv(buf, index=False)
                    buf.seek(0)
                    st.download_button("Download Recommendations (CSV)", data=buf, file_name="career_recommendations.csv", mime="text/csv")
                    st.session_state['last_recommendation'] = rec_df

            st.markdown("---")
            st.subheader("Feedback")
            if 'last_recommendation' in st.session_state:
                last = st.session_state['last_recommendation']
                career_choice = st.selectbox("Which recommended career would you like to rate?", options=list(last['Career']))
                rating = st.slider("Rate usefulness (1-5)", 1, 5, 4)
                comment = st.text_area("Optional comment")
                if st.button("Submit Feedback"):
                    save_feedback(user['id'], career_choice, int(rating), comment)
                    st.success("Thanks — your feedback was saved.")
            else:
                st.info("Get recommendations first to provide feedback.")

elif menu == "Admin":
    st.header("Admin / System Management")
    st.write("Enter admin key (demo).")
    key = st.text_input("Admin Secret Key", type="password")
    if key:
        if key == ADMIN_KEY:
            st.success("Admin authenticated.")
            st.subheader("Users")
            st.dataframe(fetch_users())
            st.subheader("Feedback")
            st.dataframe(fetch_feedback())
            st.subheader("Model")
            st.write("Train/retrain on synthetic dataset (demo).")
            if st.button("Train model (synthetic)"):
                with st.spinner("Training..."):
                    report = train_and_save_model()
                    st.success("Model trained and saved.")
                    st.code(report)
            if st.button("Download feedback CSV"):
                data = fetch_feedback()
                st.download_button("Download feedback", data=data.to_csv(index=False).encode('utf-8'), file_name="feedback.csv", mime="text/csv")
        else:
            st.error("Invalid admin key.")

# footer
st.markdown("---")
st.caption("Prototype by CareerWise AI — replace synthetic data with real labeled datasets for production. Secure credentials before deployment.")
