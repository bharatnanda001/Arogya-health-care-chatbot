# Deployment Guide - Arogya Healthcare Assistant

This guide explains how to deploy **Arogya - Smart Healthcare Assistant** to cloud platforms. Hosting Arogya in the cloud allows you and others to access it from any device 24/7.

---

## Option 1: Streamlit Community Cloud (Recommended & Free)
Streamlit Community Cloud is the easiest, fastest, and completely free way to host Streamlit applications. It syncs directly with a GitHub repository.

### Step-by-Step Instructions:
1. **Push your code to GitHub:**
   * Create a new repository on [GitHub](https://github.com/) named `Arogya`.
   * Push your local Arogya files (including `app.py`, `requirements.txt`, `healthcare_assistant_avatar.png`, and `Dockerfile`) to the repository.
   * *Note: Do not push the `chatbot_env` virtual environment folder. It is ignored by `.gitignore`.*

2. **Sign up / Sign in to Streamlit Share:**
   * Visit [Streamlit Community Cloud](https://share.streamlit.io/).
   * Click **Sign in** and authorize with your GitHub account.

3. **Deploy the App:**
   * Click **New app** on your Streamlit workspace.
   * Select your repository (`Arogya`), the branch (`main` or `master`), and set the main file path to `app.py`.
   * (Optional but recommended) Under **Advanced Settings**, choose Python Version 3.11.
   * Click **Deploy!** Your app will be live in less than 2 minutes.

4. **Add Secret Keys (Optional):**
   * If you want to use the high-fidelity cloud Gemini AI engine, go to your deployed app settings on Streamlit.
   * Locate the **Secrets** section.
   * Add your Gemini API key in TOML format:
     ```toml
     GEMINI_API_KEY = "your-actual-api-key-here"
     ```
   * The app will automatically read this key from secrets and unlock Gemini Mode for all users!

---

## Option 2: Hugging Face Spaces (Free)
Hugging Face Spaces is another excellent free hosting provider, optimized for hosting AI demos.

### Step-by-Step Instructions:
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Click **New Space** in the top right menu.
3. Name your space `Arogya`, select **Streamlit** as the Space SDK, and select **Free CPU Basic** as the hardware.
4. Set the space visibility to **Public** or **Private** and click **Create Space**.
5. Upload your files (`app.py`, `requirements.txt`, `healthcare_assistant_avatar.png`) directly through the browser UI or clone the git repository provided by Hugging Face and push your code.
6. (Optional) Go to the Space's **Settings**, scroll down to **Variables and Secrets**, and add a secret named `GEMINI_API_KEY` to configure Gemini Mode safely.

---

## Option 3: Dockerized Self-Hosting (Render, AWS, GCP, VPS)
Arogya comes with a production-ready `Dockerfile` that uses a lightweight, multi-stage build.

### local Testing with Docker:
To build and run the Docker container locally:
```bash
# Build the image
docker build -t arogya-app .

# Run the container
docker run -p 8501:8501 arogya-app
```
Then open `http://localhost:8501` in your browser.

### Deploying to Render.com:
1. Push code to GitHub.
2. Sign in to [Render](https://render.com/).
3. Click **New +** and select **Web Service**.
4. Connect your GitHub repository.
5. In settings, select **Runtime: Docker** (Render will automatically detect the `Dockerfile`).
6. Under **Environment Variables**, add:
   * Key: `GEMINI_API_KEY` | Value: `(your API key)`
7. Click **Deploy Web Service**.
