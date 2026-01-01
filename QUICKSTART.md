# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
streamlit run app.py
```

### Step 3: Use the Interface

1. Select a model (BERT or TinyLlama)
2. Enter your text
3. Adjust layer and head sliders
4. Click "Run Analysis"

## 📋 Example Use Cases

### Analyzing Sentiment

**Input**: "I love this product! It's amazing."

**What to look for**:
- High attention between "love" and "amazing"
- Token contribution showing emphasis on positive words

### Understanding Relationships

**Input**: "The cat sat on the mat"

**What to look for**:
- Attention patterns showing "cat" → "sat" → "mat" relationships
- How the model understands subject-verb-object structure

### Model Comparison

**Compare BERT vs TinyLlama**:
- Run the same text through both models
- Compare attention patterns
- Analyze differences in token contributions

## 💡 Pro Tips

1. **Start with simple sentences** to understand basic patterns
2. **Experiment with different layers** - early layers capture syntax, later layers capture semantics
3. **Use head similarity** to identify which heads are doing similar work
4. **Check entropy** to understand how focused the attention is

## 🐛 Common Issues

**Model won't load?**
- Check internet connection (first download requires it)
- Ensure you have enough disk space (~500MB per model)

**Out of memory?**
- Use shorter input texts
- Try BERT instead of TinyLlama (smaller model)

**Visualization looks wrong?**
- Check that layer/head indices are within valid ranges
- Ensure input text is not empty

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore different model architectures
- Experiment with various text inputs
- Analyze attention patterns for your specific use case

