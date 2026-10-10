
import re
import html
import joblib
import pandas as pd
import streamlit as st

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Customer Support Ticket Triage",
    page_icon="🎫",
    layout="wide"
)


# ==================================================
# 2. LOAD TRAINED MODELS
# ==================================================
@st.cache_resource
def load_models():
    queue_model = joblib.load("models/queue_model.pkl")
    queue_vectorizer = joblib.load("models/queue_vectorizer.pkl")
    priority_model = joblib.load("models/priority_model.pkl")
    priority_vectorizer = joblib.load("models/priority_vectorizer.pkl")

    return (
        queue_model,
        queue_vectorizer,
        priority_model,
        priority_vectorizer
    )


queue_model, queue_vectorizer, priority_model, priority_vectorizer = (
    load_models()
)


# ==================================================
# 3. CLEAN TICKET TEXT
# ==================================================
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove placeholders such as <name> and <email>
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove unnecessary spaces and line breaks
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==================================================
# 4. INITIALIZE SESSION HISTORY
# ==================================================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# ==================================================
# 5. INTERFACE STYLING
# ==================================================
st.markdown("""
<style>
    /* Main page */
    .stApp {
        background-color: #FAF8F9;
        color: #39313B;
    }

    
/* Add space above the app content so the title is fully visible */
.block-container {
    max-width: 1120px;
    padding-top: 2.5rem;
    padding-bottom: 1.5rem;
}


    h1, h2, h3 {
        color: #493744 !important;
    }

    p, label {
        color: #514651;
    }

    /* Header and logo */
    .app-header {
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 8px 0 18px 0;
    }

    .app-logo {
        width: 46px;
        height: 46px;
        min-width: 46px;
        border-radius: 11px;
        background: #F1DFE9;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .app-name {
        font-size: 23px;
        font-weight: 700;
        color: #493744;
    }

    .app-subtitle {
        font-size: 13px;
        color: #756574;
        margin-top: 4px;
    }

    /* Introduction panel */
    .intro-panel {
        background: #F3E8EF;
        border: 1px solid #E7D5E1;
        border-radius: 11px;
        padding: 20px 23px;
        margin: 12px 0 24px 0;
    }

    .intro-title {
        color: #493744;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .intro-description {
        color: #655565;
        font-size: 14px;
    }

    /* Ticket text area */
    [data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;
        color: #39313B !important;
        border: 1px solid #D6CAD4 !important;
        border-radius: 8px !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #817582 !important;
        opacity: 1 !important;
    }

    /* Buttons with bold, readable text */
    div[data-testid="stButton"] > button {
        background-color: #E8C5D8 !important;
        color: #3B2032 !important;
        border: 1px solid #C995B4 !important;
        border-radius: 7px !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        min-height: 42px;
        padding: 8px 20px !important;
    }

    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span,
    div[data-testid="stButton"] > button div {
        color: #3B2032 !important;
        font-weight: 800 !important;
        -webkit-text-fill-color: #3B2032 !important;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #D9A9C5 !important;
        border-color: #B77A9F !important;
    }

    /* Prediction result cards */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E4D9E2;
        border-radius: 10px;
        padding: 18px;
        min-height: 118px;
    }

    .result-label {
        font-size: 12px;
        color: #655565;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .result-value {
        font-size: 19px;
        font-weight: 700;
        margin-top: 13px;
        overflow-wrap: anywhere;
    }

    /* Human review messages */
    .status-box {
        padding: 13px 16px;
        border-radius: 8px;
        margin-top: 16px;
        font-size: 14px;
        line-height: 1.6;
    }

.status-review {
    background-color: #FCE8E8;
    color: #A12632;
    border: 1px solid #E7B7BC;
    border-left: 5px solid #C62828;
    padding: 15px;
    border-radius: 8px;
    font-weight: 500;
}

.status-ok {
    background-color: #E8F5E9;
    color: #216E39;
    border: 1px solid #B8DDBD;
    border-left: 5px solid #2E7D32;
    padding: 15px;
    border-radius: 8px;
    font-weight: 500;
}

    /* Dividers */
    hr {
        border-color: #E5DAE2;
    }

    /* One-line footer */
    .footer-note {
        color: #746875;
        font-size: 12px;
        text-align: center;
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 6. HEADER
# ==================================================
st.markdown("""
<div class="app-header">
    <div class="app-logo">🎫</div>
    <div>
        <div class="app-name">Customer Support Ticket Triage</div>
        <div class="app-subtitle">
            Support ticket classification and priority prediction
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ==================================================
# 7. INTRODUCTION
# ==================================================
st.markdown("""
<div class="intro-panel">
    <div class="intro-title">Ticket Analysis</div>
    <div class="intro-description">
        Enter a customer's issue to suggest the appropriate
        support queue and estimate its priority.
    </div>
</div>
""", unsafe_allow_html=True)


# 8. TICKET INPUT

with st.container(border=True):
    st.markdown("### Ticket details")

    ticket_text = st.text_area(
        "Describe the customer's issue",
        placeholder=(
            "Example: My payment was deducted, "
            "but my order was not confirmed."
        ),
        height=130,
        key="ticket_input"
    )

    analyze_button = st.button(
        "Analyze Ticket",
        key="analyze_ticket_button"
    )


# 9. ANALYZE TICKET
if analyze_button:

    if not ticket_text.strip():
        st.warning("Please enter a ticket description.")

    else:
        # Clean the submitted text
        cleaned_ticket = clean_text(ticket_text)

        # Convert ticket text into TF-IDF features
        queue_features = queue_vectorizer.transform(
            [cleaned_ticket]
        )

        priority_features = priority_vectorizer.transform(
            [cleaned_ticket]
        )

        # Predict the support queue
        predicted_queue = queue_model.predict(
            queue_features
        )[0]

        # Predict the priority
        predicted_priority = priority_model.predict(
            priority_features
        )[0]

        # Calculate the highest queue probability
        probabilities = queue_model.predict_proba(
            queue_features
        )[0]

        confidence = float(probabilities.max())

        # Save the latest result
        st.session_state.latest_result = {
            "queue": predicted_queue,
            "priority": predicted_priority,
            "confidence": confidence
        }

        # Add the prediction to session history
        st.session_state.prediction_history.insert(
            0,
            {
                "Ticket": " ".join(ticket_text.split())[:70],
                "Predicted Queue": predicted_queue,
                "Priority": predicted_priority.title(),
                "Confidence": f"{confidence:.1%}",
                "Human Review": (
                    "Required" if confidence < 0.50
                    else "Not flagged"
                )
            }
        )

        # Retain only the latest 20 predictions
        st.session_state.prediction_history = (
            st.session_state.prediction_history[:20]
        )


# ==================================================
# 10. PREDICTION RESULTS
# ==================================================

# ## 11. Show prediction results

# Display results only if a prediction has been saved
if st.session_state.latest_result is not None:

    # Retrieve the saved prediction dictionary
    result = st.session_state.latest_result

    # Get the confidence score before using it
    confidence = result["confidence"]

    # Display the results heading
    st.divider()
    st.markdown("## Prediction Results")
    st.caption("Suggested routing based on the submitted ticket.")

    # Create three columns for the prediction cards
    col1, col2, col3 = st.columns(3)

    # Prepare the predicted queue and priority text safely
    queue_safe = html.escape(str(result["queue"]))
    priority_safe = html.escape(str(result["priority"]).title())

    # Display the support queue card
    with col1:
        st.markdown(f"""
        <div class="result-card" style="border-top:3px solid #A88ABD;">
            <div class="result-label">Suggested Support Queue</div>
            <div class="result-value" style="color:#684B7B;">
                {queue_safe}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display the priority card
    with col2:
        priority_styles = {
            "high": ("#F8E5E8", "#983E53"),
            "medium": ("#F8EEDC", "#805B20"),
            "low": ("#E6F1E8", "#356B47"),
        }

        priority_bg, priority_fg = priority_styles.get(
            str(result["priority"]).lower(),
            ("#F1E8EF", "#71546A")
        )

        st.markdown(f"""
        <div class="result-card"
             style="background:{priority_bg};
                    border-top:3px solid {priority_fg};">
            <div class="result-label">Predicted Priority</div>
            <div class="result-value" style="color:{priority_fg};">
                {priority_safe}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display the confidence card
    with col3:
        confidence_color = (
            "#983E53" if confidence < 0.50 else "#356B47"
        )

        st.markdown(f"""
        <div class="result-card" style="border-top:3px solid #D19AB9;">
            <div class="result-label">Queue Confidence</div>
            <div class="result-value" style="color:{confidence_color};">
                {confidence:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Recommend manual review when confidence is below 50%
    if confidence < 0.50:
        st.markdown("""
        <div class="status-box status-review">
            <b>Manual review recommended</b><br>
            Queue confidence is below the 50% threshold.
            Verify the suggested queue before assigning this ticket.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="status-box status-ok">
            <b>Confidence threshold met</b><br>
            Please verify the prediction before assigning this ticket.
        </div>
        """, unsafe_allow_html=True)

    # Explain the confidence score
    st.caption(
        "Confidence is the model's highest predicted queue probability; "
        "it does not guarantee correctness."
    )

# ==================================================
# 11. RECENT PREDICTIONS — LIGHT PINK TABLE
# ==================================================
st.divider()
st.markdown("## Recent Predictions")
st.caption("Tickets analyzed during the current session.")

if st.session_state.prediction_history:

    # Convert history to a DataFrame
    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    # Escape values before inserting them into HTML
    safe_df = history_df.map(
        lambda value: html.escape(str(value))
    )

    # Convert actual records into an HTML table
    table_html = safe_df.to_html(
        index=False,
        escape=False,
        classes="history-table",
        border=0
    )

    # Style the table using light interface colours
    st.markdown("""
    <style>
/* Entire Recent Predictions table */
.history-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background-color: #F8EAF1;
    border: 1px solid #E2C5D5;
    border-radius: 10px;
    overflow: hidden;
    font-size: 14px;
}

/* Table headings */
.history-table th {
    background-color: #E8C9DB;
    color: #493044;
    padding: 14px 12px;
    text-align: left;
    font-weight: 700;
    border-bottom: 1px solid #DDB8CC;
}

/* All table cells */
.history-table td {
    background-color: #F8EAF1;
    color: #493044;
    padding: 14px 12px;
    text-align: left;
    border-bottom: 1px solid #EAD5E1;
}

/* Alternate row shade */
.history-table tbody tr:nth-child(even) td {
    background-color: #F1DDE8;
}

/* Remove the last row's bottom border */
.history-table tbody tr:last-child td {
    border-bottom: none;
}

        .history-table th:first-child {
            border-top-left-radius: 9px;
        }

        .history-table th:last-child {
            border-top-right-radius: 9px;
        }

        @media (max-width: 700px) {
            .history-table {
                font-size: 12px;
            }

            .history-table th,
            .history-table td {
                padding: 9px 7px;
                overflow-wrap: anywhere;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Display the light pink table
    st.markdown(
        table_html,
        unsafe_allow_html=True
    )

    # Clear history when requested
    if st.button("Clear History", key="clear_history_button"):
        st.session_state.prediction_history = []
        st.session_state.latest_result = None
        st.rerun()

else:
    st.info(
        "No tickets analyzed yet. Your recent predictions "
        "will appear here."
    )


# ==================================================
# 12. SINGLE-LINE FOOTER
# ==================================================
st.divider()

st.markdown("""
<div class="footer-note">
    Customer Support Ticket Triage · TF-IDF and Logistic Regression ·
    Verify uncertain predictions before routing.
</div>
""", unsafe_allow_html=True)
