import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
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

# Set page config
st.set_page_config(
    page_title="Arogya - Premium Dual-Engine Healthcare Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom Premium Styling (Dark Mode, Glassmorphism, Neon teal/cyan accents, Outfit font)
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        /* Main page backgrounds - Ultra Dark Glassmorphic */
        .stApp {
            background: radial-gradient(circle at 50% 50%, #0d1527 0%, #050811 100%);
            color: #f1f5f9 !important;
            font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
        }
        
        /* Premium sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060b18 0%, #03060c 100%) !important;
            border-right: 1px solid rgba(14, 165, 233, 0.15) !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }
        [data-testid="stSidebar"] p {
            color: #94a3b8 !important;
        }
        
        /* Typography details */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }
        
        /* Card-like premium glass panels (using Streamlit's bordered containers) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(10, 18, 36, 0.6) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-radius: 24px !important;
            border: 1px solid rgba(14, 165, 233, 0.15) !important;
            padding: 2rem !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(14, 165, 233, 0.35) !important;
            box-shadow: 0 12px 40px 0 rgba(14, 165, 233, 0.1) !important;
            transform: translateY(-2px) !important;
        }
        
        /* App header title styling */
        .app-header {
            text-align: center;
            padding: 2rem 0 1rem 0;
            margin-bottom: 1.5rem;
        }
        .app-title {
            background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 50%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            letter-spacing: -1px;
            text-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
        }
        .app-subtitle {
            color: #94a3b8;
            font-size: 1.25rem;
            font-weight: 400;
        }
        
        /* Badges */
        .engine-badge-gemini {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
            border: 1px solid rgba(14, 165, 233, 0.4);
            color: #38bdf8 !important;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.4rem 1rem;
            border-radius: 50px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
        }
        
        .engine-badge-local {
            background: linear-gradient(135deg, rgba(148, 163, 184, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
            border: 1px solid rgba(148, 163, 184, 0.3);
            color: #38bdf8 !important;
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
            background: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.25);
            color: #fca5a5;
            padding: 1.2rem 1.8rem;
            border-radius: 16px;
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.05);
        }
        
        .triage-urgent {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
            padding: 1.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(239, 68, 68, 0.1);
            margin-top: 1.5rem;
        }
        .triage-moderate {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%);
            border: 1px solid rgba(245, 158, 11, 0.4);
            color: #fbbf24;
            padding: 1.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(245, 158, 11, 0.1);
            margin-top: 1.5rem;
        }
        .triage-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
            padding: 1.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);
            margin-top: 1.5rem;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: rgba(10, 18, 36, 0.3);
            padding: 8px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 10px 24px;
            color: #94a3b8;
            font-weight: 600;
            background-color: transparent;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #38bdf8;
            background-color: rgba(14, 165, 233, 0.06);
        }
        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%) !important;
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.2);
        }
        
        /* Chat messages customizations */
        .stChatMessage {
            background-color: transparent !important;
            padding: 0px !important;
            margin-bottom: 1.8rem !important;
        }
        .stChatMessageContent {
            background: rgba(15, 26, 52, 0.6) !important;
            color: #f1f5f9 !important;
            border-radius: 20px !important;
            padding: 1.4rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
            border: 1px solid rgba(14, 165, 233, 0.15) !important;
        }
        
        /* Vitals custom metric boxes */
        .vital-metric {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 1.25rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .vital-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #38bdf8;
            margin-top: 0.25rem;
        }
        
        /* Hide Streamlit's default footer */
        footer {
            visibility: hidden;
        }
    </style>
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
- **Vomiting and diarrhea** (more common in children than adults)

#### Self-Care & Prevention Tips:
1. **Rest & Hydration:** Drink plenty of fluids (water, broth, herbal tea) and get ample rest to support your immune system.
2. **Over-the-Counter Care:** Decongestants or pain relievers (like acetaminophen or ibuprofen) can help manage symptoms under a doctor's guidance.
3. **Prevention:** The annual flu vaccine is the most effective way to prevent influenza.

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
- **General weakness and fatigue**

#### Self-Care Tips:
1. **Hydrate:** Stay well-hydrated. Sip water, clear soups, or electrolyte solutions.
2. **Cool Comfort:** Rest in a cool, well-ventilated room. Use light bedding and dress in lightweight layers.
3. **Warm Bath:** A lukewarm sponge bath can help lower body temperature safely. Avoid cold water or ice baths as they can trigger shivering and raise internal temperature.

*Red Flags: Seek professional medical care immediately if the fever exceeds 103°F (39.4°C), lasts more than three days, or is accompanied by a severe headache, stiff neck, shortness of breath, or skin rash.*""",
    
    "cough": """### Cough Guidelines & Insights
A cough is a natural reflex that clears your airway of irritants and mucus. It can be acute (lasting less than 3 weeks) or chronic.
Common causes of acute coughs include colds, influenza, sinus infections, or bronchitis.

#### Self-Care Tips:
1. **Stay Hydrated:** Warm fluids like tea with honey (for individuals over 1 year of age) can soothe an irritated throat.
2. **Humidify:** Use a cool-mist humidifier or inhale steam from a hot shower to loosen mucus.
3. **Rest:** Give your body time to heal.

*When to consult a doctor: Contact a healthcare professional if the cough lasts more than 3 weeks, is accompanied by blood, shortness of breath, high fever, or produces thick, foul-smelling mucus.*""",

    "thyroid": """### Thyroid Conditions Overview
Thyroid disorders are common conditions affecting the thyroid gland, which controls your metabolism. The most common types are:
- **Hypothyroidism (Underactive Thyroid):** Symptoms include fatigue, weight gain, dry skin, constipation, and feeling cold.
- **Hyperthyroidism (Overactive Thyroid):** Symptoms include weight loss, rapid or irregular heartbeat, anxiety, irritability, and sweating.

#### Self-Care & Management Tips:
1. **Medical Diagnostics:** A simple blood test (TSH test) is used to check how well your thyroid is working.
2. **Treatment Compliance:** If prescribed thyroid hormone replacement (e.g., levothyroxine) or anti-thyroid medication, take it exactly as directed by your endocrinologist.
3. **Regular Monitoring:** Schedule periodic follow-ups and blood tests with your healthcare provider to adjust dosages.

*Always consult a doctor or endocrinologist for proper diagnosis and treatment. Do not attempt to self-medicate or ignore persistent symptoms.*""",

    "headache": """### Headache Guidelines & Insights
Headaches are very common and can range from mild to severe. Common types include:
- **Tension Headaches:** The most common type, causing a dull, constant ache on both sides of the head.
- **Migraines:** Intense, throbbing pain, often on one side of the head, sometimes accompanied by nausea, vomiting, or sensitivity to light and sound.
- **Cluster Headaches:** Severe, recurring headaches that occur in cycles or "clusters," often around one eye.

#### Self-Care Tips:
1. **Hydration & Rest:** Drink water and rest in a quiet, dark room.
2. **Cold/Warm Compress:** Apply a cool compress to your forehead or a warm compress to your neck.
3. **Trigger Management:** Keep a headache diary to identify and avoid triggers like stress, certain foods, or lack of sleep.

*Seek immediate medical attention if you experience a sudden, extremely severe headache ("thunderclap" headache), or if it is accompanied by a fever, stiff neck, confusion, numbness, or difficulty speaking.*""",

    "cold": """### Common Cold Guidelines
The common cold is a viral infection of your nose and throat. It is usually harmless, though it may not feel that way.
Common symptoms include:
- **Runny or stuffy nose**
- **Sore throat** and **cough**
- **Congestion** and **sneezing**
- **Mild body aches** or a low-grade fever

#### Self-Care Tips:
1. **Stay Hydrated:** Drink plenty of fluids (water, warm broth, clear juices).
2. **Rest:** Allow your body time to fight off the virus.
3. **Soothe your throat:** Gargle with warm salt water or sip warm water with honey (for ages 1+).
4. **Humidify:** Use a cool-mist humidifier to ease nasal congestion.

*Consult a doctor if symptoms worsen or last longer than 10 days, or if you develop high fever, severe ear pain, or shortness of breath.*""",

    "diabetes": """### Diabetes Overview & Care
Diabetes is a chronic metabolic condition characterized by elevated levels of blood glucose (sugar).
- **Type 1 Diabetes:** The body does not produce insulin.
- **Type 2 Diabetes:** The body becomes resistant to insulin or doesn't make enough.

#### Self-Care & Management Tips:
1. **Dietary Control:** Prioritize whole grains, lean proteins, vegetables, and low-glycemic foods. Minimize simple sugars.
2. **Regular Exercise:** Physical activity increases insulin sensitivity and helps manage blood sugar levels.
3. **Routine Tracking:** Monitor blood glucose levels regularly as advised by your endocrinologist.

*Warning Signs: Seek immediate care for symptoms of extremely high sugar (ketoacidosis: breath smelling fruity, extreme thirst, rapid breathing) or low sugar (hypoglycemia: shakiness, confusion, sweating, loss of consciousness).*""",

    "hypertension": """### Hypertension (High Blood Pressure) Overview
Hypertension is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.

#### Lifestyle & Self-Care Tips:
1. **Reduce Sodium:** Limit daily salt intake (ideally under 1,500 - 2,000 mg/day).
2. **DASH Diet:** Focus on fruits, vegetables, whole grains, and low-fat dairy.
3. **Stress Relief:** Practice regular mindfulness, meditation, or light cardiovascular exercise.
4. **Consistent Monitoring:** Check your blood pressure regularly at home.

*Urgent Warning: Seek immediate emergency care if you experience a hypertensive crisis (systolic over 180 or diastolic over 120 accompanied by chest pain, shortness of breath, back pain, numbness/weakness, or difficulty speaking).*""",

    "asthma": """### Asthma Guidelines
Asthma is a chronic respiratory condition that causes your airways to narrow, swell, and produce extra mucus, making breathing difficult and triggering coughing, wheezing, and shortness of breath.

#### Self-Care & Prevention Tips:
1. **Identify Triggers:** Common triggers include pollen, dust mites, pet dander, cold air, smoke, or physical stress.
2. **Action Plan:** Have a clear asthma action plan co-created with your pulmonologist.
3. **Inhaler Compliance:** Carry your quick-relief rescue inhaler (e.g., Albuterol) everywhere and take controller medication consistently.

*Emergency signs: Seek immediate emergency care if you experience extreme difficulty breathing, your rescue inhaler provides no relief, or your fingernails or lips turn blue.*"""
}

# Cache resource to load local Model and Tokenizer directly (Resilient Offline Loader)
@st.cache_resource
def load_hf_model(model_id):
    logger.info(f"Loading local tokenizer and model for: {model_id}")
    try:
        device_map = "auto" if torch.cuda.is_available() else None
        
        # Load from cache (falls back to downloading once if not cached)
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
            # Try strictly loading from local file storage
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=True).to("cpu")
            return {"tokenizer": tokenizer, "model": model, "device": torch.device("cpu"), "status": "active"}
        except Exception as local_err:
            logger.warning(f"Strict local load failed: {str(local_err)}. Arogya will boot with dynamic offline rule engine.")
            # Graceful degradation flag
            return {"status": "degraded", "error": str(local_err)}

# Conversational Rule-Based Fallback Engine (Resilient NLP)
def generate_offline_fallback_response(question):
    clean_question = question.lower().strip()
    
    # 1. Parse for specific keywords
    for key, response in MEDICAL_KNOWLEDGE.items():
        if key in clean_question:
            return response + "\n\n*(Source: Arogya Local Curated Knowledge Base)*"
            
    # 2. General health queries fallbacks
    if any(k in clean_question for k in ["diet", "nutrition", "food", "eat"]):
        return """### General Dietary & Nutrition Tips
A healthy diet supports overall physical wellness, weight management, and immune health.
*   **Balance:** Incorporate rich lean proteins, abundant vegetables, dietary fiber, and healthy monounsaturated fats.
*   **Hydration:** Drink 2 to 3 liters of water daily.
*   **Minimize:** Reduce daily intake of highly processed sugars, excess sodium, and trans fats.
*   **Routine:** Eat consistent, well-proportioned meals rather than heavy late-night dinners.

*Disclaimer: Consult a licensed clinical dietitian for specialized meal planning.*"""
        
    if any(k in clean_question for k in ["exercise", "workout", "fitness", "run", "gym"]):
        return """### Fitness & Physical Activity Guide
Regular physical activity strengthens your cardiovascular system, improves mental health, and supports metabolism.
*   **Aero-conditioning:** Strive for 150 minutes of moderate aerobic exercise (like brisk walking) or 75 minutes of vigorous workouts weekly.
*   **Strength training:** Target all major muscle groups twice a week.
*   **Consistency:** Daily movement (even brief stretches) provides substantial long-term benefits compared to sporadic intensive training.

*Disclaimer: Obtain clinical clearance before commencing new high-intensity training regimens.*"""

    if any(k in clean_question for k in ["sleep", "insomnia", "tired"]):
        return """### Sleep Hygiene Guidelines
Adequate sleep is vital for cellular regeneration, metabolic regulation, and brain health.
*   **Schedule:** Keep a highly consistent sleep-wake schedule, even on weekends.
*   **Ambiance:** Keep your sleeping room dark, quiet, and slightly cool.
*   **Digital curfew:** Turn off screens and electronic devices at least 1 hour before bed.
*   **Avoid:** Steer clear of high caffeine and heavy meals close to sleep hours.

*Consult a doctor if insomnia or fatigue persists for more than three weeks.*"""

    # 3. Default friendly medical chatbot greeting
    return f"""### Hello! I am Arogya, your Clinical Assistant Companion.

I am currently running in **Resilient Offline NLP Mode**. 

It looks like your query ("*{question}*") didn't match our immediate high-frequency offline medical topics (Fever, Flu, Cough, Cold, Headache, Thyroid, Asthma, Diabetes, Hypertension, Diet, Exercise, Sleep).

#### How to optimize this application:
1. **Try common keywords:** Search for any of our predefined offline topics for detailed medical advice.
2. **Switch to Cloud Gemini Mode:** If you have an active internet connection, you can toggle the **Cloud Gemini Engine** in the sidebar and insert a Google Gemini API Key to unlock unlimited, state-of-the-art AI reasoning on any medical question!
3. **Download Offline Models:** If you want general offline AI generation, connect to the internet once and select a local model size in the sidebar. Arogya will download it automatically and cache it for permanent offline use!

*Please remember that I am designed for informational purposes. If you are experiencing concerning symptoms, please consult a healthcare professional.*"""

# Generate clinical response using Google Gemini API
def generate_gemini_response(question, api_key, temperature=0.6):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        system_prompt = (
            "You are Arogya, a highly professional, clinical-grade, and empathetic AI healthcare assistant. "
            "Your objective is to answer the user's healthcare question safely, clearly, and thoroughly. "
            "Use bullet points, bold text, or structured lists to make details highly readable. "
            "If the user asks about dangerous symptoms (like heavy chest pain, sudden numbness, or severe shortness of breath), "
            "strongly direct them to immediate emergency care. "
            "Provide the response directly. Do not prefix it with 'Answer:' or similar."
        )
        
        response = model.generate_content(
            contents=question,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024
            ),
            system_instruction=system_prompt
        )
        
        return response.text + "\n\n---\n*Disclaimer: This response is powered by Google Gemini. Please verify all clinical concerns with a qualified physician.*"
    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        return f"*(Error loading Gemini Brain: {str(e)})*\n\nPlease make sure your API key is correct, or switch back to Local Offline AI Mode."

# Generate medical response with multi-engine selection
def generate_medical_response(question, model_data, temperature=0.6, api_key=None, engine_mode="Offline"):
    clean_question = question.lower().strip()
    
    # 1. Cloud Gemini Mode (if selected and active)
    if engine_mode == "Cloud Gemini Premium" and api_key and HAS_GEMINI:
        return generate_gemini_response(question, api_key, temperature)
    
    # 2. Local Private Mode: First check high-precision curated local knowledge base
    for key, response in MEDICAL_KNOWLEDGE.items():
        if key in clean_question:
            return response + "\n\n*(Source: Arogya Curated Knowledge Base)*"
            
    # 3. If local Deep Learning model is successfully loaded, use it
    if model_data and model_data.get("status") == "active":
        try:
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            device = model_data["device"]
            
            prompt = f"You are a helpful clinical assistant. Provide a brief, accurate, and safe response to the medical question. Question: {question} Response:"
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
            
            safety_notice = "\n\n---\n*Disclaimer: This response is generated by a local AI model offline. Please verify any health concerns with a qualified physician.*"
            return response_text + safety_notice
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            # Fallback to local rule engine
            return generate_offline_fallback_response(question)
    
    # 4. Fallback to Dynamic Offline NLP Engine (Safeguard)
    return generate_offline_fallback_response(question)

def main():
    # Session state initialization
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        # Sidebar header with professional 3D medical chatbot avatar
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if os.path.exists("healthcare_assistant_avatar.png"):
            st.image("healthcare_assistant_avatar.png", width=120)
        st.markdown("<h2 style='margin-top: 10px; color: #38bdf8; font-size: 28px; margin-bottom: 0px;'>Arogya</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 14px; margin-top: 0px; color: #94a3b8;'>Clinical Intelligence Assistant</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
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
            # Cloud Engine Settings
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
                        <span style='color: #10b981;'>●</span> Premium Gemini Active
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="engine-badge-local" style='border-color: #f59e0b; color: #f59e0b !important;'>
                        <span style='color: #f59e0b;'>●</span> Key Required
                    </div>
                """, unsafe_allow_html=True)
        else:
            # 100% Offline Settings
            st.markdown("""
                <div class="engine-badge-local">
                    <span style='color: #10b981;'>●</span> Local Offline AI Active
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
        
        # Quick Presets
        st.markdown("### 💬 Presets for Chat")
        if st.button("What are influenza symptoms?"):
            st.session_state.temp_prompt = "Tell me about the common symptoms and self-care of the flu"
            st.rerun()
            
        if st.button("How to treat a fever?"):
            st.session_state.temp_prompt = "How is a fever treated and when should I see a doctor?"
            st.rerun()
            
        if st.button("What is thyroid disorder?"):
            st.session_state.temp_prompt = "Explain hypothyroidism and hyperthyroidism symptoms"
            st.rerun()
            
        st.markdown("---")
        if st.button("Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("<p style='text-align: center; font-size: 11px; margin-top: 20px; color: #64748b;'>Arogya v1.3.0 • Complete Dual Edition</p>", unsafe_allow_html=True)

    # Main Dashboard Header
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Arogya Healthcare Portal</h1>
            <p class="app-subtitle">Your comprehensive AI clinical companion & dynamic wellness center</p>
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
        st.markdown("<h3 style='margin-bottom: 20px; color: #38bdf8;'>💬 Consult with Clinician AI</h3>", unsafe_allow_html=True)
        
        # Load local Model with nice Streamlit spinner only if strictly in local offline mode
        model_data = None
        if engine_mode == "100% Private Offline" and selected_model_id:
            with st.spinner(f"Booting up local {model_option.split(' ')[0]} AI brain... (Downloads once, then runs offline. Fail-safe active)"):
                model_data = load_hf_model(selected_model_id)
                
            if model_data and model_data.get("status") == "degraded":
                st.warning("⚠️ Local deep learning libraries are downloading models or compiling. Active offline fallback engine has taken over. You can ask queries instantly!")
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Check for temporary prompt from presets
        query_prompt = None
        if 'temp_prompt' in st.session_state and st.session_state.temp_prompt:
            query_prompt = st.session_state.temp_prompt
            st.session_state.temp_prompt = None  # Clear
            
        chat_input = st.chat_input("Ask a medical or wellness query...")
        final_prompt = query_prompt if query_prompt else chat_input

        if final_prompt:
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            with st.chat_message("user"):
                st.markdown(final_prompt)

            with st.spinner("Arogya Clinician AI is analyzing and generating response..."):
                response = generate_medical_response(final_prompt, model_data, temperature, api_key, engine_mode)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    # ==================== TAB 2: SMART VITALS PORTAL ====================
    with tab_vitals:
        st.markdown("<h3 style='color: #38bdf8;'>📊 Interactive Health & Vitals Portal</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; margin-bottom: 30px;'>Input your physical metrics to instantly track health baselines and generate optimal wellness guides.</p>", unsafe_allow_html=True)
        
        v_col1, v_col2, v_col3 = st.columns(3)
        
        # --- BMI & Ideal Weight Calculator ---
        with v_col1:
            with st.container(border=True):
                st.markdown("<h4 style='color: #38bdf8; margin-top: 0px;'>🩺 Body Mass Index (BMI)</h4>", unsafe_allow_html=True)
                
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
                    color = "#10b981"
                    advice = "Fantastic! You are in a highly healthy weight zone. Maintain balanced nutrition and steady exercise."
                elif 25.0 <= bmi < 29.9:
                    category = "Overweight"
                    color = "#f59e0b"
                    advice = "Advisable to increase active physical play and regulate daily carbohydrate intake."
                else:
                    category = "Obesity"
                    color = "#ef4444"
                    advice = "Recommended to consult a primary care clinician to formulate a structured cardiovascular and dietary plan."
                    
                st.markdown(f"<p style='text-align: center; font-weight: 700; font-size: 1.15rem;'>Status: <span style='color: {color};'>{category}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 0.9rem; color: #cbd5e1; text-align: center;'>{advice}</p>", unsafe_allow_html=True)

        # --- Daily Water Intake Tracker ---
        with v_col2:
            with st.container(border=True):
                st.markdown("<h4 style='color: #38bdf8; margin-top: 0px;'>💧 Hydration Intelligence</h4>", unsafe_allow_html=True)
                
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
                
                # Progress bar simulation
                st.progress(min(water_need / 5.0, 1.0))
                st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; text-align: center; margin-top: 5px;'>Hydration supports metabolic function, kidney health, cognitive alertness, and physical endurance.</p>", unsafe_allow_html=True)

        # --- Daily Calories (BMR & TDEE) ---
        with v_col3:
            with st.container(border=True):
                st.markdown("<h4 style='color: #38bdf8; margin-top: 0px;'>🔥 Caloric Energy Calculator</h4>", unsafe_allow_html=True)
                
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
                
                # Custom styled responsive calories guide table
                st.markdown(f"""
                    <table style='width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem;'>
                        <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                            <td style='padding: 6px 0; color: #cbd5e1;'>Basal Metabolic Rate (BMR)</td>
                            <td style='padding: 6px 0; text-align: right; font-weight: 700; color: #38bdf8;'>{int(bmr)} kcal</td>
                        </tr>
                        <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                            <td style='padding: 6px 0; color: #cbd5e1;'>Weight Loss (Deficit)</td>
                            <td style='padding: 6px 0; text-align: right; font-weight: 700; color: #f59e0b;'>{int(tdee - 500)} kcal</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px 0; color: #cbd5e1;'>Muscle Building (Surplus)</td>
                            <td style='padding: 6px 0; text-align: right; font-weight: 700; color: #10b981;'>{int(tdee + 300)} kcal</td>
                        </tr>
                    </table>
                """, unsafe_allow_html=True)

    # ==================== TAB 3: DYNAMIC SYMPTOM TRIAGE ====================
    with tab_triage:
        st.markdown("<h3 style='color: #38bdf8;'>🩺 Clinical Symptom Triage Panel</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Tick all symptoms you are currently experiencing to evaluate potential clinical severity and recommended courses of action.</p>", unsafe_allow_html=True)
        
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
            # Red flags (immediate emergency)
            red_flags = [chest_pain, shortness_breath, stiff_neck, dizziness, loss_consciousness]
            # Moderate signs
            orange_flags = [fever_sym, stomach_pain, cough_sym, diarrhea_sym, headache_sym, body_aches]
            # Mild signs
            green_flags = [chills_sym, fatigue_sym, throat_sym, nasal_sym, nausea_sym]
            
            red_count = sum(1 for flag in red_flags if flag)
            orange_count = sum(1 for flag in orange_flags if flag)
            green_count = sum(1 for flag in green_flags if flag)
            
            # Logic evaluation
            if red_count > 0:
                st.markdown(f"""
                    <div class="triage-urgent">
                        <h3 style='margin-top:0px; color:#ef4444;'>🚨 CRITICAL TRIAGE: EMERGENCY PROTOCOL REQUIRED</h3>
                        <p style='font-size:1.1rem; line-height:1.6; font-weight: 500;'>
                            You have selected one or more <b>critical clinical red flags</b> (e.g. chest pain, shortness of breath, sudden neurological changes). 
                            These symptoms can indicate a life-threatening medical emergency.
                        </p>
                        <hr style='border-color: rgba(239, 68, 68, 0.2); margin: 1rem 0;'>
                        <h5 style='color: #ffffff; margin-bottom: 5px;'>Immediate Actions Needed:</h5>
                        <ul style='margin-top: 5px; color: #fecaca;'>
                            <li><b>Dial emergency response services (911 / 112 / 102) immediately.</b></li>
                            <li>Do not drive yourself to the hospital; await professional paramedics.</li>
                            <li>Inform someone near you of your state immediately.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif orange_count >= 2 or stomach_pain:
                st.markdown(f"""
                    <div class="triage-moderate">
                        <h3 style='margin-top:0px; color:#f59e0b;'>⚠️ MODERATE TRIAGE: CLINICAL EVALUATION SUGGESTED</h3>
                        <p style='font-size:1.1rem; line-height:1.6; font-weight: 500;'>
                            Your symptom profile indicates a <b>moderate degree of clinical stress</b>. Symptoms like stomach pain, prolonged fever, or multiple respiratory issues should be evaluated by a medical professional.
                        </p>
                        <hr style='border-color: rgba(245, 158, 11, 0.2); margin: 1rem 0;'>
                        <h5 style='color: #ffffff; margin-bottom: 5px;'>Recommended Actions:</h5>
                        <ul style='margin-top: 5px; color: #fef3c7;'>
                            <li>Contact your Primary Care Physician (PCP) or visit a local urgent care clinic within the next 24 hours.</li>
                            <li>Monitor your body temperature regularly.</li>
                            <li>Get substantial physical rest and stay well hydrated with clear fluids or electrolytes.</li>
                            <li>If symptoms worsen rapidly or you develop chest pressure/difficulty breathing, switch to emergency protocols.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            elif green_count > 0 or orange_count == 1:
                st.markdown(f"""
                    <div class="triage-low">
                        <h3 style='margin-top:0px; color:#10b981;'>💚 LOW TRIAGE: SELF-CARE & MONITORING</h3>
                        <p style='font-size:1.1rem; line-height:1.6; font-weight: 500;'>
                            Your symptom report suggests a <b>mild, localized infection or minor physical fatigue</b> (e.g. common cold, mild throat soreness). Most of these conditions resolve naturally.
                        </p>
                        <hr style='border-color: rgba(16, 185, 129, 0.2); margin: 1rem 0;'>
                        <h5 style='color: #ffffff; margin-bottom: 5px;'>Self-Care Strategy:</h5>
                        <ul style='margin-top: 5px; color: #d1fae5;'>
                            <li><b>Rest:</b> Give your body adequate sleep to boost immune response.</li>
                            <li><b>Hydration:</b> Sip water, warm herbal teas, or warm broths frequently.</li>
                            <li><b>Symptomatic Relief:</b> Use mild over-the-counter comforts under advice (e.g. warm saltwater gargles for throat, saline drops for nasal congestion).</li>
                            <li>Monitor your symptoms. If they persist for more than 7-10 days, or if high fever develops, schedule a routine clinical appointment.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No active symptoms selected. Enter your status in the checkboxes to run an instant triage assessment.")

if __name__ == "__main__":
    main()

# Upgraded Dual Edition - Developed by Antigravity AI
