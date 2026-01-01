# Groq API Setup Guide

This guide explains how to set up the Groq API for AI-generated explanations in the Transformer Explainability Lab.

## What is Groq API?

Groq provides fast inference for LLM models like LLaMA. We use it to generate human-readable explanations of attention patterns and token relationships.

## Step 1: Install Groq Package

```bash
pip install groq
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Step 2: Get Your API Key

1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up for a free account (if you don't have one)
3. Navigate to "API Keys" section
4. Click "Create API Key"
5. Copy your API key (you'll only see it once!)

## Step 3: Set the Environment Variable

### Windows (PowerShell)

**Temporary (current session only):**
```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

**Permanent (for current user):**
```powershell
[System.Environment]::SetEnvironmentVariable('GROQ_API_KEY', 'your_api_key_here', 'User')
```

### Windows (Command Prompt)

**Temporary:**
```cmd
set GROQ_API_KEY=your_api_key_here
```

**Permanent:**
```cmd
setx GROQ_API_KEY "your_api_key_here"
```

### Linux/macOS

**Temporary (current session only):**
```bash
export GROQ_API_KEY="your_api_key_here"
```

**Permanent (add to ~/.bashrc or ~/.zshrc):**
```bash
echo 'export GROQ_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Using .env File (Alternative)

1. Create a `.env` file in the project root directory
2. Add the following line:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
3. Install python-dotenv: `pip install python-dotenv`
4. The application will automatically load it (if you add dotenv support)

## Step 4: Verify Setup

Restart your Streamlit application and check if the Groq API is working:

```bash
streamlit run app.py
```

If set up correctly, you should see AI-generated explanations in the Explainability tab.

## Troubleshooting

### "Groq package not installed"
- Run: `pip install groq`
- Verify: `python -c "from groq import Groq; print('OK')"`

### "GROQ_API_KEY environment variable not set"
- Make sure you've set the environment variable correctly
- Restart your terminal/IDE after setting it
- On Windows, you may need to restart your computer for permanent changes

### "Groq API Error: ..."
- Check that your API key is correct
- Verify you have internet connection
- Check Groq API status: https://status.groq.com

## Is Groq API Required?

**No!** The Groq API is optional. The application works perfectly without it:
- All visualizations work
- Token relationship analysis works
- Coreference detection works
- Only AI-generated natural language explanations require the API

## Free Tier

Groq offers a free tier with generous limits, perfect for this application. No credit card required!

## Security Note

⚠️ **Never commit your API key to version control!**
- Don't add `.env` files to git
- Don't hardcode API keys in source code
- Use environment variables or secure secret management

