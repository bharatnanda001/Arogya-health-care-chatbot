import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import logging
import sqlite3
import os

# Try importing google-generativeai for premium Gemini support
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config (Calm Clinical Aesthetic)
st.set_page_config(
    page_title="Arogya - Privacy-First AI Health OS",
    page_icon="🩺",
    layout="wide"
)

# ----------------- SQLITE DATABASE ENGINE -----------------
def init_db():
    conn = sqlite3.connect("arogya_health.db", check_same_thread=False)
    cursor = conn.cursor()
    # Profiles table (Patient longitudinal memory)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            allergies TEXT,
            chronic_diseases TEXT,
            medications TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Vitals tracking logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vitals_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bmi REAL,
            bmi_status TEXT,
            water_target REAL,
            caloric_maintenance INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Triage logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS triage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT,
            score INTEGER,
            level TEXT,
            conditions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

# Establish persistent database connection
db_conn = init_db()

def load_profile():
    cursor = db_conn.cursor()
    cursor.execute("SELECT name, age, gender, allergies, chronic_diseases, medications FROM profiles ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        return {
            "name": row[0],
            "age": row[1],
            "gender": row[2],
            "allergies": row[3],
            "chronic_diseases": row[4],
            "medications": row[5]
        }
    return None

def save_profile(name, age, gender, allergies, chronic_diseases, medications):
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO profiles (name, age, gender, allergies, chronic_diseases, medications)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, age, gender, allergies, chronic_diseases, medications))
    db_conn.commit()

def log_vitals(bmi, status, water, calories):
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO vitals_history (bmi, bmi_status, water_target, caloric_maintenance)
        VALUES (?, ?, ?, ?)
    """, (bmi, status, water, calories))
    db_conn.commit()

def get_vitals_history():
    cursor = db_conn.cursor()
    cursor.execute("SELECT bmi, bmi_status, water_target, caloric_maintenance, created_at FROM vitals_history ORDER BY id DESC LIMIT 5")
    return cursor.fetchall()

def log_triage(symptoms, score, level, conditions):
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO triage_logs (symptoms, score, level, conditions)
        VALUES (?, ?, ?, ?)
    """, (symptoms, score, level, conditions))
    db_conn.commit()

def get_triage_history():
    cursor = db_conn.cursor()
    cursor.execute("SELECT symptoms, score, level, created_at FROM triage_logs ORDER BY id DESC LIMIT 5")
    return cursor.fetchall()

# ----------------- TRUSTED MEDICAL RAG DATABASE -----------------
TRUSTED_GUIDELINES = [
    {
        "title": "Influenza (Flu) Care Guidelines (WHO/CDC)",
        "content": "Influenza symptoms include sudden high fever, sore throat, cough, chills, body aches, and severe fatigue. Home care relies on bed rest, substantial hydration (water, electrolytes), and over-the-counter anti-inflammatories under medical advice. Annual vaccination is the primary prevention strategy. Seek emergency care immediately if shortness of breath, heavy chest pressure, or blue lips develop."
    },
    {
        "title": "Fever Management Protocol (Mayo Clinic)",
        "content": "A fever is a core body temperature elevated above 98.6°F (37°C), signifying immune fight against infection. Management includes plenty of water, cool clothing, and resting in well-ventilated temperate rooms. Warm sponge baths can lower fever safely. Avoid cold or ice baths as they cause shivering which increases internal temperature. Consult a physician if fever exceeds 103°F (39.4°C), lasts over 3 days, or is accompanied by a severe headache or stiff neck."
    },
    {
        "title": "Asthma Management & Trigger Guide (CDC/GINA)",
        "content": "Asthma causes chronic airway inflammation, narrowing, coughing, wheezing, and shortness of breath. Common triggers include pollen, air pollution, cold weather, dust mites, pet dander, and respiratory viral infections. Patients must maintain access to rescue inhalers (like Albuterol) and take prescription daily controller inhalers. Seek emergency ER care immediately if a rescue inhaler provides no relief or extreme respiratory distress is present."
    },
    {
        "title": "Diabetes Mellitus Care (MedlinePlus/ADA)",
        "content": "Diabetes is a metabolic disease characterized by chronic hyperglycemia. Management is focused on low-glycemic, high-fiber dietary intakes, regular active aerobic exercise (150 mins weekly), weight control, and absolute compliance with prescribed insulin or oral medications (Metformin). Check blood sugar levels regularly. Extreme hypoglycemia (shakiness, cold sweat, confusion) requires rapid sugar intake. Hyperglycemia can lead to dangerous ketoacidosis."
    },
    {
        "title": "Hypertension (High Blood Pressure) Guidelines (AHA/ACC)",
        "content": "Hypertension is defined as blood pressure consistently above 130/80 mmHg. Lifestyle modifications are critical: lower sodium consumption (under 1500mg daily), focus on the DASH diet (high in vegetables, fruits, whole grains), reduce alcohol, and maintain moderate aerobic physical exercise. Monitor blood pressure daily. A hypertensive crisis (systolic over 180 or diastolic over 120 with chest pain or vision changes) is a medical emergency."
    },
    {
        "title": "Thyroid Disorders Care Sheet (MedlinePlus)",
        "content": "Thyroid disorders disrupt standard metabolism. Hypothyroidism (underactive) symptoms include fatigue, cold intolerance, dry skin, and weight gain, treated with levothyroxine hormone replacement taken on empty stomach. Hyperthyroidism (overactive) symptoms include anxiety, tremors, weight loss, and rapid heartbeat. Regular TSH blood tests are required to monitor hormone levels and adjust therapeutic dosages."
    },
    {
        "title": "Acute & Chronic Cough Triage Guidelines (ACC)",
        "content": "Coughing is an essential reflex that clears airways of mucus and irritants. Acute coughs (under 3 weeks) are commonly due to viral colds or bronchitis. Chronic coughs (over 8 weeks) require clinical investigation for GERD, asthma, or post-nasal drip. Warm fluids, honey (for ages over 1), and hydration soothe dry throats. Consult a doctor if coughing yields blood, thick foul sputum, or extreme shortness of breath."
    },
    {
        "title": "Common Cold Guidance (CDC)",
        "content": "The common cold is a mild, self-limiting viral infection of the upper respiratory tract. Symptoms include runny/stuffy nose, sneezing, mild sore throat, and low fever. Relief relies on resting, drinking warm broths or water, and saline nasal sprays. Do not take antibiotics as they are ineffective against viruses. Symptoms resolve in 7-10 days; consult a clinic if symptoms worsen or persist past 10 days."
    }
]

@st.cache_resource
def build_rag_index():
    logger.info("Initializing SentenceTransformer and building FAISS vector index...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{g['title']}: {g['content']}" for g in TRUSTED_GUIDELINES]
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return {"model": model, "index": index, "texts": texts, "status": "active"}
    except Exception as e:
        logger.error(f"Failed to build RAG index: {str(e)}")
        return {"status": "disabled", "error": str(e)}

def retrieve_rag_context(query, rag_data):
    if not rag_data or rag_data.get("status") != "active":
        return ""
    try:
        model = rag_data["model"]
        index = rag_data["index"]
        texts = rag_data["texts"]
        
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, 2)
        
        retrieved_texts = []
        for i in indices[0]:
            if i < len(texts):
                retrieved_texts.append(texts[i])
        
        return "\n\n".join(retrieved_texts)
    except Exception as e:
        logger.error(f"RAG retrieval error: {str(e)}")
        return ""

# ----------------- SOOTHING CLINICAL DESIGN SYSTEM -----------------
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        /* Main page backgrounds - Soft Slate Clinical Look */
        .stApp {
            background: #f8fafc;
            color: #1e293b !important;
            font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
        }
        
        /* Force dark text for all checkboxes, radios, selectboxes, and widget inputs */
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] p,
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] p,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] p,
        div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] p,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stNumberInput"] p,
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextInput"] p,
        div[data-testid="stTextArea"] label,
        div[data-testid="stTextArea"] p,
        label[data-testid="stWidgetLabel"] p {
            color: #1e293b !important;
        }
        
        /* Ensure normal markdown texts inside body also render dark */
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            color: #334155 !important;
        }
        
        /* Soothing white clinical sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #0f172a !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
            color: #475569 !important;
        }
        
        /* Typography details */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            color: #0f172a;
            font-weight: 600;
        }
        
        /* Card-like premium glass panels (using Streamlit's bordered containers) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border-radius: 20px !important;
            border: 1px solid #e2e8f0 !important;
            padding: 1.8rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(14, 165, 233, 0.35) !important;
            box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.1) !important;
            transform: translateY(-1px) !important;
        }
        
        /* App header title styling */
        .app-header {
            text-align: center;
            padding: 1.5rem 0 1rem 0;
            margin-bottom: 1rem;
        }
        .app-title {
            background: linear-gradient(135deg, #0f172a 0%, #0369a1 50%, #0d9488 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            letter-spacing: -1px;
        }
        .app-subtitle {
            color: #64748b;
            font-size: 1.2rem;
            font-weight: 400;
        }
        
        /* Badges */
        .engine-badge-gemini {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #16a34a !important;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.4rem 1rem;
            border-radius: 50px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        .engine-badge-local {
            background-color: #f0f9ff;
            border: 1px solid #bae6fd;
            color: #0284c7 !important;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.4rem 1rem;
            border-radius: 50px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        /* Alert panels for disclaimers and triage */
        .medical-disclaimer {
            background: #fff5f5;
            border: 1px solid #fee2e2;
            color: #b91c1c;
            padding: 1.2rem 1.8rem;
            border-radius: 16px;
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
        }
        
        .triage-urgent {
            background: #fff5f5;
            border: 1px solid #fecaca;
            color: #dc2626;
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(220, 38, 38, 0.05);
            margin-top: 1.5rem;
        }
        .triage-moderate {
            background: #fffbeb;
            border: 1px solid #fef3c7;
            color: #d97706;
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(217, 119, 6, 0.05);
            margin-top: 1.5rem;
        }
        .triage-low {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #16a34a;
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(22, 101, 52, 0.05);
            margin-top: 1.5rem;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: #ffffff;
            padding: 6px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 20px;
            color: #64748b;
            font-weight: 600;
            background-color: transparent;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #0284c7;
            background-color: #f0f9ff;
        }
        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: linear-gradient(135deg, #0ea5e9 0%, #0d9488 100%) !important;
            box-shadow: 0 4px 10px rgba(14, 165, 233, 0.15);
        }
        
        /* Chat messages customizations */
        .stChatMessage {
            background-color: transparent !important;
            padding: 0px !important;
            margin-bottom: 1.5rem !important;
        }
        .stChatMessageContent {
            background: #ffffff !important;
            color: #1e293b !important;
            border-radius: 16px !important;
            padding: 1.2rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        /* Vitals custom metric boxes */
        .vital-metric {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 1.25rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .vital-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0ea5e9;
            margin-top: 0.25rem;
        }
        
        /* Explainability Reasoning Panel styling */
        .explainability-panel {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
            border-radius: 16px;
            margin-top: 1.5rem;
            border-left: 5px solid #0ea5e9;
        }
        
        /* Voice control components */
        .voice-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 1rem;
        }
        
        /* Hide Streamlit's default footer */
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# 🎙️ HTML5 Native Web Speech API JavaScript Injector
st.markdown("""
    <script>
    function startDictation() {
        if (window.hasOwnProperty('webkitSpeechRecognition')) {
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = "en-US";
            recognition.start();
            
            var micBtn = document.getElementById('mic-btn');
            micBtn.style.color = '#ef4444';
            micBtn.innerHTML = '🎙️ Listening...';
            
            recognition.onresult = function(e) {
                var text = e.results[0][0].transcript;
                recognition.stop();
                
                // Copy transcribed text to clipboard
                navigator.clipboard.writeText(text);
                
                var display = document.getElementById('speech-output');
                display.innerHTML = '<strong>Spoken (Copied to Clipboard):</strong> "' + text + '"';
                micBtn.style.color = '#0ea5e9';
                micBtn.innerHTML = '🎙️ Tap to Speak Symptom';
            };
            
            recognition.onerror = function(e) {
                recognition.stop();
                micBtn.style.color = '#0ea5e9';
                micBtn.innerHTML = '🎙️ Tap to Speak Symptom';
            };
        } else {
            alert("Speech recognition is not supported in this browser. Please use Chrome or Edge.");
        }
    }
    
    function speakText(textId) {
        var textElement = document.getElementById(textId);
        if (textElement) {
            var text = textElement.innerText;
            window.speechSynthesis.cancel(); // Cancel any ongoing speech
            var utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        }
    }
    </script>
""", unsafe_allow_html=True)

# Predefined medical knowledge base for common queries (Offline fallback)
MEDICAL_KNOWLEDGE = {
    "flu": """### Influenza (Flu) Overview
Common influenza symptoms typically include:
- **High fever** or feeling feverish/chills
- **Cough** and **sore throat**
- **Runny or stuffy nose**
- **Muscle or body aches**
- **Headaches**
- **Fatigue** (extreme tiredness)

#### Self-Care & Prevention Tips:
1. **Rest & Hydration:** Drink plenty of fluids (water, broth, herbal tea) and get ample rest to support your immune system.
2. **Over-the-Counter Care:** Decongestants or pain relievers (like acetaminophen or ibuprofen) can help manage symptoms under a doctor's guidance.

*Seek immediate medical attention if you experience difficulty breathing, chest pain, persistent dizziness, confusion, or a fever that doesn't respond to medication.*""",
    
    "fever": """### Fever Overview
A fever is a temporary increase in your body temperature, often due to an illness. It is a sign that your body is fighting off an infection.
Common symptoms accompanying a fever include:
- **Elevated body temperature** (above 98.6°F / 37°C)
- **Chills and shivering**
- **Sweating**
- **Headache**
- **Muscle aches**
- **Loss of appetite**
- **Dehydration**

#### Self-Care Tips:
1. **Hydrate:** Stay well-hydrated. Sip water, clear soups, or electrolyte solutions.
2. **Cool Comfort:** Rest in a cool, well-ventilated room. Use light bedding and dress in lightweight layers.

*Red Flags: Seek professional medical care immediately if the fever exceeds 103°F (39.4°C), lasts more than three days, or is accompanied by a severe headache, stiff neck, shortness of breath, or skin rash.*""",
    
    "cough": """### Cough Guidelines & Insights
A cough is a natural reflex that clears your airway of irritants and mucus. It can be acute (lasting less than 3 weeks) or chronic.
Common causes of acute coughs include colds, influenza, sinus infections, or bronchitis.

#### Self-Care Tips:
1. **Stay Hydrated:** Warm fluids like tea with honey (for individuals over 1 year of age) can soothe an irritated throat.
2. **Humidify:** Use a cool-mist humidifier or inhale steam from a hot shower to loosen mucus.

*When to consult a doctor: Contact a healthcare professional if the cough lasts more than 3 weeks, is accompanied by blood, shortness of breath, high fever, or produces thick, foul-smelling mucus.*"""
}

# Cache resource to load local Model and Tokenizer directly (Offline fail-safe)
@st.cache_resource
def load_hf_model(model_id):
    logger.info(f"Loading local tokenizer and model for: {model_id}")
    try:
        device_map = "auto" if torch.cuda.is_available() else None
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float32,
            local_files_only=False
        )
        
        if device_map is None:
            model = model.to("cpu")
            
        return {"tokenizer": tokenizer, "model": model, "device": model.device, "status": "active"}
    except Exception as e:
        logger.error(f"First-pass load failed: {str(e)}. Checking strict local files...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=True).to("cpu")
            return {"tokenizer": tokenizer, "model": model, "device": torch.device("cpu"), "status": "active"}
        except Exception as local_err:
            logger.warning(f"Strict local load failed: {str(local_err)}. Arogya will boot with dynamic offline rule engine.")
            return {"status": "degraded", "error": str(local_err)}

# Dynamic Local NLP Fallback Engine
def generate_offline_fallback_response(question):
    clean_question = question.lower().strip()
    
    for key, response in MEDICAL_KNOWLEDGE.items():
        if key in clean_question:
            return response + "\n\n*(Source: Arogya Local Curated Knowledge Base)*"
            
    if any(k in clean_question for k in ["diet", "nutrition", "food", "eat"]):
        return """### General Dietary & Nutrition Tips
A healthy diet supports overall physical wellness, weight management, and immune health.
*   **Balance:** Incorporate rich lean proteins, abundant vegetables, dietary fiber, and healthy monounsaturated fats.
*   **Hydration:** Drink 2 to 3 liters of water daily.
*   **Minimize:** Reduce daily intake of highly processed sugars, excess sodium, and trans fats.

*Disclaimer: Consult a licensed clinical dietitian for specialized meal planning.*"""
        
    if any(k in clean_question for k in ["exercise", "workout", "fitness", "run", "gym"]):
        return """### Fitness & Physical Activity Guide
Regular physical activity strengthens your cardiovascular system, improves mental health, and supports metabolism.
*   **Cardio:** Strive for 150 minutes of moderate aerobic exercise (like brisk walking) weekly.
*   **Strength:** Target all major muscle groups twice a week.

*Disclaimer: Obtain clinical clearance before commencing new high-intensity training regimens.*"""

    return f"""### Hello! I am Arogya, your Clinical Assistant Companion.

I am currently running in **Resilient Offline NLP Mode**.

It looks like your query ("*{question}*") didn't match our immediate high-frequency offline medical topics (Fever, Flu, Cough, Diet, Exercise).

#### How to optimize this application:
1. **Try common keywords:** Search for any of our predefined offline topics for detailed medical advice.
2. **Switch to Cloud Gemini Mode:** If you have an active internet connection, you can toggle the **Cloud Gemini Engine** in the sidebar and insert a Google Gemini API Key to unlock unlimited, state-of-the-art AI reasoning on any medical question!
3. **Download Offline Models:** If you want general offline AI generation, connect to the internet once and select a local model size in the sidebar. Arogya will download it automatically and cache it for permanent offline use!"""

# Generate clinical response using Google Gemini API
def generate_gemini_response(question, api_key, temperature=0.6, profile_ctx="", rag_ctx=""):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""You are Arogya, a highly professional, clinical-grade, and empathetic AI healthcare assistant.
Answer the user's healthcare question safely, clearly, and thoroughly.

{profile_ctx}

Trusted Clinical References (Answer strictly using this grounded evidence. If unrelated, rely on safe medical consensus):
{rag_ctx}

User's Question: {question}

Response (use bullet points, bold text, or structured lists to make details highly readable):"""
        
        response = model.generate_content(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024
            )
        )
        
        return response.text + "\n\n---\n*Disclaimer: This response is powered by Google Gemini and grounded in trusted medical guidelines. Please verify all clinical concerns with a qualified physician.*"
    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        return f"*(Error loading Gemini Brain: {str(e)})*\n\nPlease make sure your API key is correct, or switch back to Local Offline AI Mode."

# Generate medical response with multi-engine selection & RAG + Memory injection
def generate_medical_response(question, model_data, temperature=0.6, api_key=None, engine_mode="100% Private Offline", rag_data=None):
    # 1. Retrieve trusted clinical guidelines context (RAG)
    rag_context = ""
    if rag_data:
        rag_context = retrieve_rag_context(question, rag_data)
        
    # 2. Retrieve longitudinal health memory (SQLite)
    profile = load_profile()
    profile_context = ""
    if profile:
        profile_context = (
            f"Patient Background Memory:\n"
            f"- Patient Name: {profile['name']}\n"
            f"- Age: {profile['age']} | Gender: {profile['gender']}\n"
            f"- Documented Allergies: {profile['allergies']}\n"
            f"- Chronic Conditions: {profile['chronic_diseases']}\n"
            f"- Active Medications: {profile['medications']}\n"
        )
        
    # 3. Cloud Gemini Engine Mode
    if engine_mode == "Cloud Gemini Premium" and api_key and HAS_GEMINI:
        return generate_gemini_response(question, api_key, temperature, profile_context, rag_context)
    
    # 4. Local Private Mode: First check high-precision curated local knowledge base
    clean_question = question.lower().strip()
    for key, response in MEDICAL_KNOWLEDGE.items():
        if key in clean_question:
            return response + "\n\n*(Source: Arogya Curated Knowledge Base)*"
            
    # 5. If local Deep Learning model is successfully loaded, use it (injecting context)
    if model_data and model_data.get("status") == "active":
        try:
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            device = model_data["device"]
            
            prompt = f"""You are Arogya, a helpful clinical assistant.
Use this patient context and clinical references to answer.
{profile_context}
References: {rag_context}
Question: {question}
Response:"""
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=250,
                    min_length=25,
                    temperature=temperature,
                    repetition_penalty=1.15,
                    do_sample=True if temperature > 0.05 else False
                )
                
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = response_text.replace("Answer:", "").replace("Medical answer:", "").strip()
            
            safety_notice = "\n\n---\n*Disclaimer: This response is generated by a local AI model offline and grounded in local guidelines. Please verify any health concerns with a qualified physician.*"
            return response_text + safety_notice
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            return generate_offline_fallback_response(question)
    
    # 6. Fallback to Dynamic Offline NLP Engine (Safeguard)
    return generate_offline_fallback_response(question)

def main():
    # Session state initialization
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Load RAG FAISS index dynamically
    rag_data = build_rag_index()
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if os.path.exists("healthcare_assistant_avatar.png"):
            st.image("healthcare_assistant_avatar.png", width=100)
        st.markdown("<h2 style='margin-top: 10px; color: #0284c7; font-size: 26px; margin-bottom: 0px;'>Arogya</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 13px; margin-top: 0px; color: #64748b;'>AI Health Operating System</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # 👤 Health Profile Engine (SQLite)
        st.markdown("### 👤 Health Profile & Memory")
        profile = load_profile()
        
        # Load defaults
        p_name = profile["name"] if profile else "John Doe"
        p_age = int(profile["age"]) if profile else 30
        p_gender = profile["gender"] if profile else "Male"
        p_allergies = profile["allergies"] if profile else "None"
        p_chronic = profile["chronic_diseases"] if profile else "None"
        p_meds = profile["medications"] if profile else "None"
        
        with st.expander("Update Clinical File", expanded=False):
            new_name = st.text_input("Full Name", value=p_name)
            new_age = st.number_input("Age", min_value=1, max_value=120, value=p_age)
            new_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(p_gender))
            new_allergies = st.text_area("Allergies (e.g., Penicillin)", value=p_allergies)
            new_chronic = st.text_area("Chronic Conditions (e.g., Asthma)", value=p_chronic)
            new_meds = st.text_area("Active Medications", value=p_meds)
            
            if st.button("Save Health Profile"):
                save_profile(new_name, new_age, new_gender, new_allergies, new_chronic, new_meds)
                st.success("Health profile updated inside SQLite!")
                st.rerun()
                
        # Display active profile details
        if profile:
            st.markdown(f"""
                <div style='background-color:#f8fafc; border:1px solid #e2e8f0; padding:10px; border-radius:10px; font-size:12px; color:#334155; line-height:1.5;'>
                    <strong>Profile:</strong> {profile['name']} ({profile['age']}, {profile['gender']})<br>
                    <strong>Allergies:</strong> {profile['allergies']}<br>
                    <strong>Chronic:</strong> {profile['chronic_diseases']}<br>
                    <strong>Meds:</strong> {profile['medications']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No active profile detected. Fill out the form above to initialize your local longitudinal memory.")
            
        st.markdown("---")
        
        # Dual-Engine AI Selection
        st.markdown("### 🧠 AI Engine Mode")
        engine_mode = st.radio(
            "Select Intelligence Source",
            options=["100% Private Offline", "Cloud Gemini Premium"],
            index=0,
            help="100% Private Offline runs entirely on your CPU/GPU without internet. Cloud Gemini provides supercharged reasoning using secure APIs."
        )
        
        api_key = ""
        model_option = ""
        selected_model_id = None
        
        if engine_mode == "Cloud Gemini Premium":
            default_key = os.environ.get("GEMINI_API_KEY", "")
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                value=default_key,
                help="Insert your Gemini API key. Responses are processed securely on your local browser session."
            )
            
            if not HAS_GEMINI:
                st.error("⚠️ The 'google-generativeai' package is not installed. Please launch the app using 'run.bat' to install it automatically or run 'pip install google-generativeai' in your environment.")
            
            if api_key and HAS_GEMINI:
                st.markdown("""
                    <div class="engine-badge-gemini">
                        <span style='color: #16a34a;'>●</span> Premium Gemini Active
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="engine-badge-local" style='border-color: #d97706; color: #d97706 !important;'>
                        <span style='color: #d97706;'>●</span> Key Required
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="engine-badge-local">
                    <span style='color: #0284c7;'>●</span> Local Offline AI Active
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            model_option = st.selectbox(
                "Local AI Brain Size",
                options=[
                    "Small (Fastest, ~300MB)",
                    "Base (Balanced, ~990MB)",
                    "Large (Detailed, ~3.13GB)"
                ],
                index=0,
                help="Small loads instantly. Large has the best medical reasoning but requires significant memory."
            )
            model_map = {
                "Small (Fastest, ~300MB)": "google/flan-t5-small",
                "Base (Balanced, ~990MB)": "google/flan-t5-base",
                "Large (Detailed, ~3.13GB)": "google/flan-t5-large"
            }
            selected_model_id = model_map[model_option]
            if model_option == "Large (Detailed, ~3.13GB)":
                st.sidebar.warning("⚠️ Flan-T5-Large requires ~3.1GB RAM. If your system has limited memory, loading may take several minutes or fail. Small/Base sizes are recommended for quick local execution.")
            
        temperature = st.slider(
            "Confidence / Creativity",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Lower values are highly precise, strict and clinical; higher values are more expressive."
        )
        
        st.markdown("---")
        if st.button("Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("<p style='text-align: center; font-size: 11px; margin-top: 20px; color: #64748b;'>Arogya v1.4.0 • Enterprise OS Edition</p>", unsafe_allow_html=True)

    # Main Dashboard Header
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Arogya Health OS</h1>
            <p class="app-subtitle">Privacy-First AI Health Operating System & Evidence-Grounded Wellness Portal</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="medical-disclaimer">
            <strong>⚠️ Medical Triage Disclaimer:</strong> Arogya is built solely for preliminary information, wellness monitoring, and educational purposes. 
            The diagnostic feedback and calculators provided here are <strong>not substitutes for professional medical advice, clinical examinations, or emergency hospital treatment</strong>. 
            If you are experiencing severe, life-threatening symptoms, immediately dial emergency services (911 / 112) or go to the nearest emergency room.
        </div>
    """, unsafe_allow_html=True)

    # Tabs structure
    tab_chat, tab_vitals, tab_triage = st.tabs([
        "💬 Clinical Consultation", 
        "📊 Smart Vitals Portal", 
        "🩺 Interactive Symptom Triage"
    ])

    # ==================== TAB 1: AI CLINICAL CONSULTATION ====================
    with tab_chat:
        st.markdown("<h3 style='margin-bottom: 10px; color: #0284c7;'>💬 Consult with Clinician AI</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; margin-top: 0px;'>Queries are processed privately and grounded in verified WHO, CDC, and Mayo Clinic references.</p>", unsafe_allow_html=True)
        
        # Load local Model with nice Streamlit spinner only if strictly in local offline mode
        model_data = None
        if engine_mode == "100% Private Offline" and selected_model_id:
            with st.spinner(f"Booting up local {model_option.split(' ')[0]} AI brain... (Downloads once, then runs offline. Fail-safe active)"):
                model_data = load_hf_model(selected_model_id)
                
            if model_data and model_data.get("status") == "degraded":
                st.warning("⚠️ Local deep learning libraries are downloading models or compiling. Active offline fallback engine has taken over. You can ask queries instantly!")
        
        # 🎙️ Hands-Free Voice Control Board
        st.markdown("""
            <div class="voice-card">
                <button id="mic-btn" onclick="startDictation()" style="background:transparent; border:none; color:#0ea5e9; font-weight:600; cursor:pointer;">
                    🎙️ Tap to Speak Symptom
                </button>
                <span style="color:#e2e8f0;">|</span>
                <span id="speech-output" style="color:#64748b; font-size:13px;">(Spoken transcriptions copy automatically to your clipboard to paste below)</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Display chat messages from history
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(f"<div id='msg-text-{idx}'>{message['content']}</div>", unsafe_allow_html=True)
                
                # Add browser TTS speaker button for assistant responses
                if message["role"] == "assistant":
                    st.markdown(f"""
                        <button onclick="speakText('msg-text-{idx}')" style="background:#f1f5f9; border:1px solid #e2e8f0; border-radius:8px; padding:4px 8px; font-size:12px; color:#475569; cursor:pointer; margin-top:8px;">
                            🔊 Speak Response
                        </button>
                    """, unsafe_allow_html=True)

        chat_input = st.chat_input("Ask a medical or wellness query...")

        if chat_input:
            st.session_state.messages.append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.markdown(chat_input)

            with st.spinner("Arogya Clinician AI is analyzing and generating response..."):
                response = generate_medical_response(chat_input, model_data, temperature, api_key, engine_mode, rag_data)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    # ==================== TAB 2: SMART VITALS PORTAL ====================
    with tab_vitals:
        st.markdown("<h3 style='color: #0284c7;'>📊 Interactive Health & Vitals Portal</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; margin-bottom: 30px;'>Input your physical metrics to instantly track health baselines and generate optimal wellness guides.</p>", unsafe_allow_html=True)
        
        v_col1, v_col2, v_col3 = st.columns(3)
        
        # --- BMI & Ideal Weight Calculator ---
        with v_col1:
            with st.container(border=True):
                st.markdown("<h4 style='color: #0284c7; margin-top: 0px;'>🩺 Body Mass Index (BMI)</h4>", unsafe_allow_html=True)
                
                unit_system = st.radio("System of Units", ["Metric (cm, kg)", "Imperial (in, lbs)"], key="bmi_units")
                
                if unit_system == "Metric (cm, kg)":
                    height = st.number_input("Height (in cm)", min_value=50, max_value=250, value=170, step=1, key="metric_h")
                    weight = st.number_input("Weight (in kg)", min_value=10, max_value=250, value=70, step=1, key="metric_w")
                    bmi = weight / ((height / 100) ** 2)
                else:
                    height = st.number_input("Height (in inches)", min_value=20, max_value=100, value=67, step=1, key="imp_h")
                    weight = st.number_input("Weight (in pounds)", min_value=20, max_value=500, value=154, step=1, key="imp_w")
                    bmi = (weight / (height ** 2)) * 703
                    
                st.markdown(f"<div class='vital-metric'>Your BMI Value<div class='vital-value'>{bmi:.1f}</div></div>", unsafe_allow_html=True)
                
                # Clinical interpretation
                if bmi < 18.5:
                    category = "Underweight"
                    color = "#60a5fa"
                    advice = "Consider speaking with a physician or nutritionist. Focus on nutrient-dense foods to safely build mass."
                elif 18.5 <= bmi < 24.9:
                    category = "Normal Weight"
                    color = "#16a34a"
                    advice = "Fantastic! You are in a highly healthy weight zone. Maintain balanced nutrition and steady exercise."
                elif 25.0 <= bmi < 29.9:
                    category = "Overweight"
                    color = "#d97706"
                    advice = "Advisable to increase active physical play and regulate daily carbohydrate intake."
                else:
                    category = "Obesity"
                    color = "#ef4444"
                    advice = "Recommended to consult a primary care clinician to formulate a structured cardiovascular and dietary plan."
                    
                st.markdown(f"<p style='text-align: center; font-weight: 700; font-size: 1.15rem;'>Status: <span style='color: {color};'>{category}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 0.9rem; color: #475569; text-align: center;'>{advice}</p>", unsafe_allow_html=True)

        # --- Daily Water Intake Tracker ---
        with v_col2:
            with st.container(border=True):
                st.markdown("<h4 style='color: #0284c7; margin-top: 0px;'>💧 Hydration Intelligence</h4>", unsafe_allow_html=True)
                
                user_weight_water = st.number_input("Your Weight (kg)", min_value=20, max_value=200, value=70, key="water_weight")
                activity_level = st.select_slider(
                    "Daily Physical Activity",
                    options=["Sedentary", "Moderate Exercise", "Heavy Athlete"],
                    value="Moderate Exercise"
                )
                climate = st.selectbox("Your Local Climate", ["Cool / Air Conditioned", "Temperate / Moderate", "Hot / Humid"])
                
                # Basic hydration requirement: 35ml per kg of bodyweight
                water_need = user_weight_water * 0.035
                
                # Adjust for activity
                if activity_level == "Moderate Exercise":
                    water_need += 0.5
                elif activity_level == "Heavy Athlete":
                    water_need += 1.0
                    
                # Adjust for climate
                if climate == "Hot / Humid":
                    water_need += 0.5
                elif climate == "Cool / Air Conditioned":
                    water_need -= 0.2
                
                glasses = water_need / 0.250 # 250ml glass
                
                st.markdown(f"<div class='vital-metric'>Optimal Daily Water<div class='vital-value'>{water_need:.2f} L</div></div>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 1.05rem; font-weight: 500;'>Equivalent to approximately <b>{int(glasses)} glasses</b> (250ml each) per day.</p>", unsafe_allow_html=True)
                
                st.progress(min(water_need / 5.0, 1.0))
                st.markdown("<p style='font-size: 0.85rem; color: #64748b; text-align: center; margin-top: 5px;'>Hydration supports metabolic function, kidney health, cognitive alertness, and physical endurance.</p>", unsafe_allow_html=True)

        # --- Daily Calories (BMR & TDEE) ---
        with v_col3:
            with st.container(border=True):
                st.markdown("<h4 style='color: #0284c7; margin-top: 0px;'>🔥 Caloric Energy Calculator</h4>", unsafe_allow_html=True)
                
                age = st.number_input("Age (Years)", min_value=1, max_value=120, value=25)
                gender = st.radio("Biological Gender", ["Male", "Female"], horizontal=True)
                c_height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, key="cal_h")
                c_weight = st.number_input("Weight (kg)", min_value=10, max_value=250, value=70, key="cal_w")
                c_activity = st.selectbox(
                    "Weekly Workouts",
                    [
                        "Little or no exercise (Sedentary)",
                        "Light exercise (1-3 days/week)",
                        "Moderate exercise (3-5 days/week)",
                        "Heavy exercise (6-7 days/week)"
                    ]
                )
                
                # Mifflin-St Jeor Equation
                if gender == "Male":
                    bmr = (10 * c_weight) + (6.25 * c_height) - (5 * age) + 5
                else:
                    bmr = (10 * c_weight) + (6.25 * c_height) - (5 * age) - 161
                    
                # Activity multiplier
                multipliers = {
                    "Little or no exercise (Sedentary)": 1.2,
                    "Light exercise (1-3 days/week)": 1.375,
                    "Moderate exercise (3-5 days/week)": 1.55,
                    "Heavy exercise (6-7 days/week)": 1.725
                }
                tdee = bmr * multipliers[c_activity]
                
                st.markdown(f"<div class='vital-metric'>Maintenance Calories<div class='vital-value'>{int(tdee)} kcal</div></div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <table style='width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; color:#475569;'>
                        <tr style='border-bottom: 1px solid #e2e8f0;'>
                            <td style='padding: 6px 0; color: #475569;'>Basal Metabolic Rate (BMR)</td>
                            <td style='padding: 6px 0; text-align: right; font-weight: 700; color: #0284c7;'>{int(bmr)} kcal</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #e2e8f0;'>
                            <td style='padding: 6px 0; color: #475569;'>Weight Loss (Deficit)</td>
                            <td style='padding: 6px 0; text-align: right; font-weight: 700; color: #d97706;'>{int(tdee - 500)} kcal</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px 0; color: #475569;'>Muscle Building (Surplus)</td>
                            <td style='padding: 6px 0; text-align: right; font-weight: 700; color: #16a34a;'>{int(tdee + 300)} kcal</td>
                        </tr>
                    </table>
                """, unsafe_allow_html=True)
                
                # Log to SQLite
                if st.button("Save Vitals Baseline"):
                    log_vitals(bmi, category, water_need, int(tdee))
                    st.success("Physical vitals baseline logged persistently in SQLite!")
                    st.rerun()
                    
        # Vitals longitudinal memory history visualization
        st.markdown("---")
        st.markdown("#### 📜 Longitudinal Vitals History (Saved in SQLite)")
        history = get_vitals_history()
        if history:
            for h in history:
                st.markdown(f"**[{h[4].split(' ')[0]}]** BMI: `{h[0]:.1f}` ({h[1]}) | Daily Water target: `{h[2]:.2f} L` | Caloric maintenance threshold: `{h[3]} kcal`")
        else:
            st.info("No prior vitals baseline records logged yet.")

    # ==================== TAB 3: DYNAMIC SYMPTOM TRIAGE ====================
    with tab_triage:
        st.markdown("<h3 style='color: #0284c7;'>🩺 Clinical Symptom Triage Panel</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; margin-bottom: 25px;'>Tick all symptoms you are currently experiencing to evaluate potential clinical severity and recommended courses of action.</p>", unsafe_allow_html=True)
        
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            st.markdown("#### ⚡ Critical Symptoms (Immediate Warning Signs)")
            chest_pain = st.checkbox("Severe Chest Pain or Pressure")
            shortness_breath = st.checkbox("Severe Shortness of Breath / Difficulty Breathing")
            stiff_neck = st.checkbox("Stiff Neck accompanied by High Fever & Confusion")
            dizziness = st.checkbox("Sudden Speech Impairment, Face Drooping or Severe Dizziness")
            loss_consciousness = st.checkbox("Fainting or Brief Loss of Consciousness")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🌡️ General Systemic Symptoms")
            fever_sym = st.checkbox("High Body Temperature (Fever)")
            chills_sym = st.checkbox("Chills and Shivering")
            fatigue_sym = st.checkbox("Extreme Fatigue or General Body Weakness")
            body_aches = st.checkbox("Generalized Muscle or Joint Aches")
            
        with t_col2:
            st.markdown("#### 🗣️ Respiratory & ENT Symptoms")
            cough_sym = st.checkbox("Persistent or Spasmodic Cough")
            throat_sym = st.checkbox("Sore Throat / Painful Swallowing")
            nasal_sym = st.checkbox("Runny or Severely Stuffy Nose")
            headache_sym = st.checkbox("Moderate to Severe Headache")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🤢 Gastrointestinal Symptoms")
            nausea_sym = st.checkbox("Nausea or Vomiting")
            diarrhea_sym = st.checkbox("Frequent Diarrhea")
            stomach_pain = st.checkbox("Acute Stomach or Abdominal Pain")
            
        # Triage Evaluation Button
        st.markdown("---")
        triage_btn = st.button("🚨 Run Triage Assessment", type="primary", use_container_width=True)
        
        if triage_btn:
            # Score symptoms
            score = 0
            checked_symptoms = []
            
            # Critical (Weight 4-5)
            if chest_pain:
                score += 5
                checked_symptoms.append("Severe Chest Pain (+5)")
            if shortness_breath:
                score += 4
                checked_symptoms.append("Difficulty Breathing (+4)")
            if stiff_neck:
                score += 4
                checked_symptoms.append("Stiff Neck + High Fever (+4)")
            if dizziness:
                score += 5
                checked_symptoms.append("Sudden Speech Impairment / Dizziness (+5)")
            if loss_consciousness:
                score += 4
                checked_symptoms.append("Fainting / Loss of Consciousness (+4)")
                
            # Moderate (Weight 2-3)
            if fever_sym:
                score += 2
                checked_symptoms.append("High Body Temperature (+2)")
            if stomach_pain:
                score += 3
                checked_symptoms.append("Acute Abdominal Pain (+3)")
            if cough_sym:
                score += 2
                checked_symptoms.append("Persistent Cough (+2)")
            if diarrhea_sym:
                score += 2
                checked_symptoms.append("Frequent Diarrhea (+2)")
            if headache_sym:
                score += 2
                checked_symptoms.append("Moderate to Severe Headache (+2)")
                
            # Mild (Weight 1)
            if chills_sym:
                score += 1
                checked_symptoms.append("Chills and Shivering (+1)")
            if fatigue_sym:
                score += 1
                checked_symptoms.append("Extreme Fatigue (+1)")
            if throat_sym:
                score += 1
                checked_symptoms.append("Sore Throat (+1)")
            if nasal_sym:
                score += 1
                checked_symptoms.append("Runny/Stuffy Nose (+1)")
            if nausea_sym:
                score += 1
                checked_symptoms.append("Nausea/Vomiting (+1)")
                
            # Determine Triage Level
            if score >= 8:
                risk_level = "HIGH RISK / EMERGENCY PROTOCOL"
                css_class = "triage-urgent"
                possible_match = "Cardiovascular Event, Severe Respiratory Failure, or Meningitis warning signs."
                clinical_rationale = "You have checked one or more high-weight clinical warning signs. A high risk score index signifies a severe, time-critical physiological threat."
                instructions = """
                <ul>
                    <li><b>Dial emergency services (911 / 112 / 102) immediately.</b></li>
                    <li>Do not drive yourself; await paramedics.</li>
                    <li>Inform family, friends, or neighbors of your status immediately.</li>
                </ul>
                """
            elif 4 <= score <= 7:
                risk_level = "MODERATE RISK / CLINICAL CONSULTATION RECOMMENDED"
                css_class = "triage-moderate"
                possible_match = "Systemic Viral Infection, Severe Gastroenteritis, or Acute Bronchitis."
                clinical_rationale = "Your checklist maps to multiple moderate symptoms acting concurrently, which raises systemic physiological stress."
                instructions = """
                <ul>
                    <li>Schedule a medical consultation with your primary physician or visit an urgent care center within 24 hours.</li>
                    <li>Rest completely, hydrate extensively, and monitor body temperatures.</li>
                    <li>If warning signs (chest pain or difficulty breathing) develop, activate Emergency protocols.</li>
                </ul>
                """
            elif 1 <= score <= 3:
                risk_level = "LOW RISK / SELF-CARE & MONITORING"
                css_class = "triage-low"
                possible_match = "Upper Respiratory Cold, Minor Fatigue, or Transient Rhinovirus infection."
                clinical_rationale = "Symptoms are localized, low-density, and non-critical, representing typical self-limiting infections."
                instructions = """
                <ul>
                    <li>Get plenty of rest and 8 hours of sleep.</li>
                    <li>Sip warm teas, electrolyte-rich water, or broths frequently.</li>
                    <li>Monitor warning signs. If symptoms persist for more than 7-10 days, contact a physician.</li>
                </ul>
                """
            else:
                risk_level = "NO RISK / NORMAL BASELINE"
                css_class = "triage-low"
                possible_match = "Healthy Baseline status."
                clinical_rationale = "No symptoms active."
                instructions = "Maintain active fitness and balanced hydration."
                
            # Render risk scoring alert
            st.markdown(f"""
                <div class="{css_class}">
                    <h3 style='margin-top:0px; color:inherit;'>📋 TRIAGE LEVEL: {risk_level}</h3>
                    <p style='font-size:1.1rem; line-height:1.6; font-weight: 500; color:inherit;'>
                        Total Clinical Triage Risk Index: <b>{score} / 15</b>
                    </p>
                    <hr style='border-color: rgba(0,0,0,0.05); margin: 1rem 0;'>
                    <h5 style='color: inherit; margin-bottom: 5px;'>Triage Instructions:</h5>
                    {instructions}
                </div>
            """, unsafe_allow_html=True)
            
            # Render 🩺 AI Clinical Reasoning & Explainability Panel
            st.markdown(f"""
                <div class="explainability-panel">
                    <h4 style="color:#0284c7; margin-top:0px;">🩺 AI Clinical Reasoning Panel</h4>
                    <p style="font-size:0.95rem; line-height:1.5; color:#334155;">
                        <strong>Indicated Clinical Matches:</strong> {possible_match}<br>
                        <strong>Diagnostic Confidence:</strong> {"High (Highly Specific)" if score >= 8 else "Moderate (Suggestive)" if score >= 4 else "Low (General Baseline)"}<br><br>
                        <strong>Underlying Rationale:</strong><br>
                        {clinical_rationale}<br><br>
                        <strong>Checked Symptoms Density (Weights):</strong><br>
                        {", ".join(checked_symptoms) if checked_symptoms else "None"}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Log to SQLite
            log_triage(", ".join(checked_symptoms), score, risk_level, possible_match)
            st.success("Triage diagnostic check logged persistently inside SQLite health database!")
            
        # Triage logs history visualization
        st.markdown("---")
        st.markdown("#### 📜 Longitudinal Symptom Triage History (Saved in SQLite)")
        triage_history = get_triage_history()
        if triage_history:
            for t in triage_history:
                st.markdown(f"**[{t[3].split(' ')[0]}]** Triage Level: `{t[2]}` | Score: `{t[1]}/15` | Symptoms: *{t[0]}*")
        else:
            st.info("No prior symptom triage logs saved yet.")

if __name__ == "__main__":
    main()

# Enterprise AI Health OS - Engineered by Antigravity AI
