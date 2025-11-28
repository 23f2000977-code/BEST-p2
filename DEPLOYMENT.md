# Deploying Hybrid Agent to Hugging Face Spaces

## Option 1: Docker Space (Recommended for File Creation)

### Step 1: Create Dockerfile

Create `Dockerfile` in `hybrid_sharing/`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright dependencies
RUN apt-get update && apt-get install -y \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_hybrid.txt .
RUN pip install --no-cache-dir -r requirements_hybrid.txt

# Install Playwright browsers
RUN playwright install chromium

# Copy application files
COPY . .

# Create directory for file operations
RUN mkdir -p /app/hybrid_llm_files

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "hybrid_main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Step 2: Create .env Template

Create `.env.example` in `hybrid_sharing/`:

```bash
# Gemini API Keys (add as many as you have)
GOOGLE_API_KEY=your_gemini_key_1
GOOGLE_API_KEY_2=your_gemini_key_2
GOOGLE_API_KEY_3=your_gemini_key_3
GOOGLE_API_KEY_4=your_gemini_key_4

# OpenAI Fallback
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

# Quiz Credentials
TDS_EMAIL=your_email@example.com
TDS_SECRET=your_secret_key
```

### Step 3: Update hybrid_main.py for HF Spaces

Add at the top of `hybrid_main.py`:

```python
import os

# For Hugging Face Spaces, read secrets from environment
# These will be set in Space settings
if not os.path.exists('.env'):
    # Running on HF Spaces - secrets are in environment variables
    pass
else:
    # Running locally - load from .env file
    load_dotenv()
```

### Step 4: Deploy to Hugging Face

1. **Create a new Space:**
   - Go to https://huggingface.co/new-space
   - Choose "Docker" as the SDK
   - Select hardware (CPU Basic is fine for testing, upgrade if needed)

2. **Upload files:**
   ```bash
   cd hybrid_sharing
   git init
   git add .
   git commit -m "Initial commit"
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```

3. **Set environment variables in Space Settings:**
   - Go to your Space → Settings → Variables and secrets
   - Add each API key as a secret:
     - `GOOGLE_API_KEY`
     - `GOOGLE_API_KEY_2`
     - `GOOGLE_API_KEY_3`
     - `GOOGLE_API_KEY_4`
     - `OPENAI_API_KEY`
     - `OPENAI_MODEL`
     - `TDS_EMAIL`
     - `TDS_SECRET`

4. **Wait for build:**
   - HF will build your Docker image
   - This may take 5-10 minutes

5. **Test the deployment:**
   ```bash
   curl -X POST https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/quiz \
     -H "Content-Type: application/json" \
     -d '{"email":"test@test.com","secret":"123","url":"https://quiz-url.com/q1.html"}'
   ```

---

## Option 2: Gradio Space (Simpler but Limited)

### Step 1: Create app.py

Create `app.py` in `hybrid_sharing/`:

```python
import gradio as gr
import requests
import threading
from hybrid_main import app
import uvicorn

# Start FastAPI server in background
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=start_server, daemon=True).start()

# Gradio interface
def submit_quiz(email, secret, url):
    response = requests.post(
        "http://localhost:8000/quiz",
        json={"email": email, "secret": secret, "url": url}
    )
    return response.json()

iface = gr.Interface(
    fn=submit_quiz,
    inputs=[
        gr.Textbox(label="Email"),
        gr.Textbox(label="Secret", type="password"),
        gr.Textbox(label="Quiz URL")
    ],
    outputs=gr.JSON(label="Response"),
    title="Hybrid Quiz Solver",
    description="Submit a quiz URL to solve it automatically"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
```

### Step 2: Deploy

1. Create Space with "Gradio" SDK
2. Upload files
3. Set environment variables in Settings
4. Wait for deployment

---

## Important Notes:

### File Creation on HF Spaces:

⚠️ **Files created during runtime are temporary**
- `hybrid_llm_files/` will be created but cleared on restart
- Logs will be lost unless you implement persistent storage
- For persistent storage, you need:
  - Docker Space (not Gradio)
  - Persistent volume mounted to `/data`
  - Paid tier

### Recommended for Exam:

**Run locally** during the exam for:
- ✅ Guaranteed file access
- ✅ No deployment issues
- ✅ Full control
- ✅ Logs preserved

**Use HF Space** for:
- ✅ Testing before exam
- ✅ Backup option
- ✅ Remote access if needed

---

## Quick Deploy Script

Create `deploy_to_hf.sh`:

```bash
#!/bin/bash

# Configuration
SPACE_NAME="your-space-name"
HF_USERNAME="your-username"

# Create Dockerfile if it doesn't exist
if [ ! -f Dockerfile ]; then
    echo "Creating Dockerfile..."
    # Dockerfile content here
fi

# Initialize git if needed
if [ ! -d .git ]; then
    git init
    git add .
    git commit -m "Initial deployment"
fi

# Add HF remote
git remote add space https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME 2>/dev/null || true

# Push to HF
git push space main

echo "Deployed to https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo "Don't forget to set environment variables in Space Settings!"
```

Make it executable:
```bash
chmod +x deploy_to_hf.sh
```

---

## Testing Checklist:

After deployment, test:
- [ ] Health check: `GET /healthz`
- [ ] Simple question (Q1-Q5)
- [ ] API question (Q6-Q7)
- [ ] Visualization question (Q8)
- [ ] File download question (Q9)
- [ ] Complex question (Q10+)
- [ ] Check logs are being created
- [ ] Verify key rotation works
- [ ] Test OpenAI fallback

---

**For the exam, I still recommend running locally for maximum reliability!**
