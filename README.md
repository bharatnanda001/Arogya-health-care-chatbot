# Arogya - Privacy-First AI Health OS 🩺

**Arogya** is an enterprise-grade, privacy-first **AI Health Operating System** and clinical wellness portal. It leverages local deep learning models (Flan-T5) and an offline semantic vector database (FAISS) to deliver a 100% private, evidence-grounded medical consultation experience completely offline. It also supports a cloud-based hybrid fallback to Google's state-of-the-art Gemini API for advanced medical reasoning.

Designed to be a comprehensive clinical companion, Arogya preserves patient security through localized SQLite database memory, integrates hands-free browser voice interactions, and features a symptom triage engine with an interactive **AI Clinical Reasoning & Explainability Panel**.

---

## 🚀 Key Product Architecture & Features

### 1. Dual-Engine Conversational AI
* **100% Private Offline Mode:** Runs completely local Hugging Face deep learning models (Google Flan-T5 family) directly on consumer CPUs. 
* **Cloud Gemini Premium Mode:** Leverages `gemini-1.5-flash` for rich, empathetic clinical conversations under low latencies.
* **Resilient NLP Keyword Fallback:** Intercepts operations seamlessly if models are loading, providing zero-downtime curated answers from an offline dictionary.

### 2. Evidence-Grounded RAG (Retrieval-Augmented Generation)
* Grounded strictly in validated clinical guidelines from **WHO, CDC, and the Mayo Clinic**.
* Converts trusted guidelines into dense, 384-dimensional semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
* Queries an in-memory **FAISS (Facebook AI Similarity Search)** index in real-time, retrieving the top relevant guides to inject as grounded context into the prompt, preventing hallucinations.

### 3. Persistent Longitudinal Memory (SQLite)
* Integrated with a local serverless SQLite relational database (`arogya_health.db`).
* Persistently records and manages three primary core schemas:
  * **`profiles`:** Stores patient age, gender, chronic history (e.g., asthma, diabetes), allergies, and medication baselines.
  * **`vitals_history`:** Stores multi-parameter tracking entries from the vitals portal.
  * **`triage_logs`:** Logs historical triage queries to allow physicians to track long-term trends.
* Injects active patient baselines into the conversational prompt, enabling persistent contextual memory (e.g., *"You reported asthma last week; your current cough could be triggered by it..."*).

### 4. Weighted Symptom Triage & Explainability
* Implements a **Weighted Clinical Risk Index** (0 to 15 points) mapping systemic, respiratory, critical, and GI indicators.
* Checks for high-weight clinical warning signs (chest pain, shortness of breath, sudden speech loss) to immediately flag high-risk **Red Alert Emergency Protocols** directing patients to urgent emergency services.
* Renders an **AI Clinical Reasoning Panel** detailing:
  * **Indicated Clinical Matches**
  * **Diagnostic Confidence** (High, Moderate, Low)
  * **Underlying Physiological Rationale**
  * **Symptom Densities and Weights**

### 5. Hands-Free Voice AI Board
* Integrates client-side **Web Speech API** natively via JavaScript.
* **Speech-to-Text (STT):** Dictate clinical symptoms hands-free directly in-browser. The transcribed text is copied automatically to the user's clipboard for seamless submission.
* **Text-to-Speech (TTS):** Plays synthesized spoken audio reading assistant responses aloud.

### 6. Clean Slate-Light Clinical UX Theme
* Replaces complex cyberpunk designs with a premium, soothing clinical layout inspired by leading telemedicine brands.
* Built using Google Fonts `Outfit` and `Inter`.
* Features custom container borders (`div[data-testid="stVerticalBlockBorderWrapper"]`) converted into modern glassmorphic clinic cards with micro-hover translations.

---

## 🛠️ High-Level Technical Stack

* **Frontend Framework:** Python Streamlit (v1.57.0) with HTML5/CSS3 and JS injections.
* **Vector Search Database:** FAISS CPU Index (`faiss-cpu`) & Hugging Face Sentence-Transformers.
* **Database Engine:** SQLite Relational Engine (`sqlite3`).
* **Deep Learning Framework:** PyTorch & Hugging Face Transformers (`AutoModelForSeq2SeqLM`, `AutoTokenizer`).
* **Cloud Integration:** Google Generative AI Python SDK (`google-generativeai`).
* **Voice Synthesis & Recognition:** Web Speech API (`webkitSpeechRecognition` & `SpeechSynthesisUtterance`).
* **Containerization:** Multi-stage production `Dockerfile`.

---

## 📦 Quick Start & Local Setup

### Direct Windows Setup (One-Click)
We have included a launch helper `run.bat`. Simply double-click **`run.bat`** in your File Explorer! 

The script will automatically check/activate your virtual environment, install missing packages (like `sentence-transformers` and `faiss-cpu`), and launch the portal in your web browser.

### Manual Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bharatnanda001/Arogya-health-care-chatbot.git
   cd Arogya
   ```

2. **Initialize and Activate Virtual Environment:**
   ```bash
   python -m venv chatbot_env
   # On Windows:
   chatbot_env\Scripts\activate
   # On macOS/Linux:
   source chatbot_env/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Health OS App:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 🛡️ Medical Disclaimer
**Arogya** is designed for educational, preliminary health assessment, and wellness monitoring purposes only. The diagnostic scoring models and calculators are not substitutes for professional clinical diagnostics, medical advice, or urgent emergency care. If you are experiencing severe, life-threatening symptoms, immediately dial local emergency services (911 / 112 / 102) or go to the nearest emergency room.

---

## 📄 License
This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.
