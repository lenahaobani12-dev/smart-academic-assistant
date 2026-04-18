import streamlit as st
import pandas as pd
import re
import string
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Smart Academic Assistant",
    page_icon="🎓",
    layout="wide"
)

# =========================
# Session State Init
# =========================
if "task_list" not in st.session_state:
    st.session_state.task_list = []  # list of dicts: {text, label, days, date}

# =========================
# Helper Functions
# =========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = text.strip()
    return text


def build_dataset() -> pd.DataFrame:
    text_data = {
        "text": [
            # Task
            "Finish math assignment",
            "Complete programming project",
            "Write research paper",
            "Prepare presentation for class",
            "Submit lab report",
            "Read chapter 5",
            "Solve physics problems",
            "Work on group project",
            "Complete online quiz",
            "Review lecture notes",
            "Do my homework",
            "Finish homework",
            "Complete assignment tonight",
            "Study for class",
            "Do my project work",
            "Finish AI assignment",
            "Submit project",
            "Work on final report",
            "Prepare class notes",
            "Do the lab task",

            # Exam
            "Final exam tomorrow",
            "Midterm exam next week",
            "Calculus exam on Monday",
            "Physics test this Friday",
            "Chemistry final exam",
            "English exam preparation",
            "Programming exam practice",
            "Statistics test",
            "Biology exam review",
            "History final test",
            "Exam in 2 days",
            "Quiz tomorrow",
            "I have a test next week",
            "Study for the final exam",
            "Upcoming midterm exam",
            "Test on Friday",
            "Prepare for the quiz",
            "Revision for exam",
            "Exam preparation today",
            "Practice exam questions",

            # Reminder
            "Submit assignment by 5pm",
            "Meeting with advisor",
            "Register for next semester",
            "Pay tuition fee",
            "Buy textbooks",
            "Deadline for project proposal",
            "Attend workshop",
            "Library books due",
            "Scholarship application deadline",
            "Group meeting at 3pm",
            "Do not forget the deadline",
            "Remember to submit the task",
            "Meeting tomorrow morning",
            "Project due tonight",
            "Assignment deadline today",
            "Reminder to register for courses",
            "Don't forget to pay fees",
            "Send email to professor",
            "Check portal for grades",
            "Reminder about group meeting",

            # Other
            "Hello how are you",
            "What time is lunch",
            "Weather is nice today",
            "Call mom later",
            "Watch movie tonight",
            "Gym session at 6pm",
            "Buy groceries",
            "Weekend plans",
            "Check social media",
            "Listen to music",
            "Hangout with friends",
            "Go out with friends",
            "Meet my friends",
            "Chill with friends",
            "Spend time with friends"
        ],
        "label": (
            ["Task"] * 20 +
            ["Exam"] * 20 +
            ["Reminder"] * 20 +
            ["Other"] * 15
        )
    }

    df = pd.DataFrame(text_data)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


@st.cache_resource
def train_model():
    df = build_dataset()

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    return model, vectorizer, metrics, df


def predict_text(text: str, model, vectorizer) -> str:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]


def count_tasks(task_list):
    task_count    = sum(1 for t in task_list if t["label"] == "Task")
    exam_count    = sum(1 for t in task_list if t["label"] == "Exam")
    reminder_count = sum(1 for t in task_list if t["label"] == "Reminder")
    other_count   = sum(1 for t in task_list if t["label"] == "Other")
    return task_count, exam_count, reminder_count, other_count


def calculate_stress(task_count: int, exam_count: int, min_days_to_exam: int) -> str:
    if exam_count > 0 and min_days_to_exam <= 3:
        return "High"
    if task_count >= 4:
        return "High"
    if task_count >= 2 or exam_count > 0:
        return "Medium"
    return "Low"


def stress_rules_engine(stress_level: str, min_days_to_exam: int, task_count: int, reminder_count: int):
    alerts = []
    suggestions = []

    if min_days_to_exam <= 3:
        alerts.append(f"⚠️ An exam is very close ({min_days_to_exam} day(s) away).")
        suggestions.append("Start revising immediately and focus on high-priority topics.")

    if stress_level == "High":
        alerts.append("🚨 High stress detected.")
        suggestions.append("Take short breaks and reduce overload where possible.")

    elif stress_level == "Medium":
        alerts.append("⚠️ Moderate stress level.")
        suggestions.append("Manage your time carefully and organize your priorities.")

    if task_count >= 5:
        alerts.append("📚 You have too many tasks.")
        suggestions.append("Break tasks into smaller steps and prioritize urgent ones.")

    if reminder_count > 0:
        alerts.append(f"🔔 You have {reminder_count} reminder(s) — don't miss any deadlines.")
        suggestions.append("Check your reminders and mark them on your calendar.")

    if not alerts:
        suggestions.append("✅ No major risk detected. Keep following your current study plan.")

    return alerts, suggestions


# =========================
# Train Once
# =========================
model, vectorizer, metrics, df = train_model()

# =========================
# UI Header
# =========================
st.title("🎓 Smart Academic Assistant")
st.markdown(
    """
This tool analyzes student academic inputs using **AI-based text classification**
and generates **stress-aware alerts and suggestions**.

**What it does:**
- Classifies each input as **Task / Exam / Reminder / Other**
- Accumulates all added tasks across the session
- Estimates academic stress level based on all tasks combined
- Provides alerts and recommendations
"""
)

# =========================
# Main Layout
# =========================
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("➕ Add a Task")

    new_input = st.text_input(
        "Enter one task or note:",
        placeholder="e.g. Final exam tomorrow"
    )
    days_for_this_task = st.slider("Days to Exam (for this task)", 0, 30, 5)

    add_clicked = st.button("Add Task", use_container_width=True)

    if add_clicked:
        text = new_input.strip()
        if not text:
            st.warning("Please enter a task before clicking Add.")
        else:
            label = predict_text(text, model, vectorizer)
            st.session_state.task_list.append({
                "text": text,
                "label": label,
                "days": days_for_this_task,
                "date": str(date.today())
            })
            st.success(f"Added: **{text}** → `{label}`")

    # Show current task list
    if st.session_state.task_list:
        st.subheader("📋 Your Task List")
        display_df = pd.DataFrame(st.session_state.task_list).rename(columns={
            "text": "Task",
            "label": "Label",
            "days": "Days to Exam",
            "date": "Date Added"
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        col_analyze, col_clear = st.columns([1, 1])
        with col_analyze:
            analyze_clicked = st.button("🔍 Analyze Stress", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("🗑️ Clear All Tasks", use_container_width=True)

        if clear_clicked:
            st.session_state.task_list = []
            st.rerun()
    else:
        analyze_clicked = False
        st.info("No tasks added yet. Enter a task above and click **Add Task**.")

with right_col:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    st.metric("Precision", f"{metrics['precision']:.2f}")
    st.metric("Recall", f"{metrics['recall']:.2f}")
    st.metric("F1 Score", f"{metrics['f1']:.2f}")

    with st.expander("Dataset Information"):
        st.write(f"Total training examples: **{len(df)}**")
        st.write("Classes: **Task, Exam, Reminder, Other**")

    with st.expander("Quick Example Inputs"):
        st.code(
            "Final exam tomorrow\nSubmit lab report\nMeeting with advisor\nBuy groceries",
            language="text"
        )

# =========================
# Analysis Section
# =========================
if analyze_clicked and st.session_state.task_list:
    task_list = st.session_state.task_list
    task_count, exam_count, reminder_count, other_count = count_tasks(task_list)

    # Use minimum days among exam tasks; fall back to all tasks
    exam_tasks = [t for t in task_list if t["label"] == "Exam"]
    if exam_tasks:
        min_days = min(t["days"] for t in exam_tasks)
    else:
        min_days = min(t["days"] for t in task_list)

    stress_level = calculate_stress(task_count, exam_count, min_days)
    alerts, suggestions = stress_rules_engine(stress_level, min_days, task_count, reminder_count)

    st.divider()

    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:
        st.subheader("📊 Predicted Labels")
        results_df = pd.DataFrame({
            "Input Text": [t["text"] for t in task_list],
            "Predicted Label": [t["label"] for t in task_list],
            "Days to Exam": [t["days"] for t in task_list],
            "Date Added": [t["date"] for t in task_list],
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    with result_col2:
        st.subheader("🧠 Stress Summary")
        st.write(f"**Task Count:** {task_count}")
        st.write(f"**Exam Count:** {exam_count}")
        st.write(f"**Reminder Count:** {reminder_count}")
        st.write(f"**Other Count:** {other_count}")
        st.write(f"**Closest Exam:** {min_days} day(s)")
        st.write(f"**Stress Level:** {stress_level}")

    st.subheader("🚨 Alerts")
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.info("No alerts generated.")

    st.subheader("💡 Suggestions")
    for suggestion in suggestions:
        st.success(suggestion)

# =========================
# Footer
# =========================
st.divider()
st.caption("Built for AI Application Programming project - Smart Academic Reminder and Stress Monitor")
