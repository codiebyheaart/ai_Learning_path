# ai_Learning_path# AI/LLM Learning Path - Practical Guide
*Learn by doing - concepts and exercises without spoon-feeding code*

---

## 1. Understanding Transformers Architecture

### What to Learn:
- **Self-Attention Mechanism** - how models "focus" on different parts of input
- **Embeddings** - converting text to numbers
- **Multi-Head Attention** - parallel attention layers

### Why It Matters:
Understanding how transformers work helps you:
- Debug model behavior
- Choose the right model for your task
- Optimize performance
- Understand limitations (context windows, attention patterns)

### Exercise to Try:
```
1. Load a pre-trained model (BERT or GPT-2)
2. Input: "The cat sat on the mat"
3. Extract embeddings for each word
4. Check the shape - understand dimensions
5. Try different sentences, compare embeddings
```

**Key Insight**: Similar words have similar embeddings. Try "king" vs "queen" vs "car" - measure distances.

---

## 2. Prompt Engineering & Chain-of-Thought

### What to Learn:
- **Zero-shot** - asking without examples
- **Few-shot** - providing examples
- **Chain-of-Thought** - asking model to think step-by-step
- **System prompts** - setting context/role

### Why It Matters:
Better prompts = better outputs. Can improve accuracy by 30-50% without any fine-tuning.

### Exercise to Try:
```
Same task: "Calculate 15% tip on $47.50"

Try 3 approaches:
1. Direct: Just ask the question
2. Few-shot: Give 2 examples first
3. CoT: Ask "let's solve step by step"

Compare quality and accuracy
```

**Key Insight**: Adding "think step by step" often improves reasoning tasks dramatically.

---

## 3. RAG Systems (Retrieval Augmented Generation)

### What to Learn:
- **Embeddings** for semantic search
- **Vector similarity** (cosine similarity)
- **Retrieval** before generation
- **Context injection** into prompts

### Why It Matters:
- LLMs have knowledge cutoff
- RAG adds real-time data without retraining
- Most common enterprise AI pattern
- Powers chatbots, docs Q&A, customer support

### Exercise to Try:
```
Create knowledge base:
doc1 = "Python is easy to learn"
doc2 = "JavaScript runs in browsers"  
doc3 = "Rust is memory safe"

Steps:
1. Convert docs to embeddings (use sentence-transformers)
2. User asks: "What language is good for web?"
3. Convert question to embedding
4. Find most similar doc (cosine similarity)
5. Send retrieved doc + question to LLM
```

**Key Insight**: Retrieval gives LLM specific context - it doesn't need to memorize everything.

---

## 4. Fine-tuning & PEFT (LoRA, QLoRA)

### What to Learn:
- **Full fine-tuning** vs **PEFT** (Parameter-Efficient Fine-Tuning)
- **LoRA** - adding small trainable matrices
- **Rank** - controls how many parameters to train
- **Adapters** - modular fine-tuning

### Why It Matters:
- Full fine-tuning is expensive (need to update billions of parameters)
- LoRA trains <1% of parameters, saves 90% GPU memory
- Can fine-tune on consumer hardware
- Industry standard now

### Exercise to Try:
```
Understand the math:
- Original layer: 1024 × 1024 = 1,048,576 parameters
- LoRA with rank=8: (1024×8) + (8×1024) = 16,384 parameters
- That's 64x fewer parameters!

Try:
1. Load GPT-2 (small model)
2. Count original parameters
3. Apply LoRA with rank=4, 8, 16
4. Compare trainable parameters
```

**Key Insight**: LoRA adds "W = W_original + A×B" where A and B are small matrices. Original weights stay frozen.

---

## 5. Vector Databases & Semantic Search

### What to Learn:
- **Vector storage** (not just SQL anymore)
- **Approximate Nearest Neighbor (ANN)** search
- **Metadata filtering** (combine semantic + filters)
- **Different DBs**: ChromaDB, Pinecone, Weaviate, FAISS

### Why It Matters:
- Traditional search = keyword matching (misses meaning)
- Semantic search = meaning-based (understands "car" ≈ "automobile")
- Essential for RAG, recommendation systems, image search
- Scales to millions of vectors

### Exercise to Try:
```
Use ChromaDB:

1. Create collection
2. Add documents:
   - "Machine learning with neural networks"
   - "Deep learning for computer vision"
   - "Python programming basics"
   
3. Query: "AI image recognition"
4. See which doc is closest (should be #2)

5. Try metadata: add {"language": "python", "year": 2024}
6. Search with filter: only 2024 docs
```

**Key Insight**: Vector DB automatically converts text to embeddings. You just store and search.

---

## 6. LLM Evaluation & Benchmarking

### What to Learn:
- **Accuracy metrics** (exact match, F1 score)
- **BLEU/ROUGE** for text generation
- **Human evaluation** (still important!)
- **Benchmarks**: MMLU, HumanEval, GSM8K, HellaSwag

### Why It Matters:
- Can't improve what you don't measure
- Compare different models/prompts objectively
- Know when model is good enough for production
- Track regression when updating

### Exercise to Try:
```
Create mini-benchmark:

Questions with expected answers:
Q1: "2+2=?" → Expected: "4"
Q2: "Capital of France?" → Expected: "Paris"
Q3: "Who wrote Hamlet?" → Expected: "Shakespeare"

Test your LLM:
1. Send questions
2. Check if expected answer in response
3. Calculate accuracy: correct/total
4. Try different prompts, measure improvement
```

**Key Insight**: Real benchmarks use thousands of questions across domains. Start small, then use public benchmarks.

---

## Popular Benchmarks Reference

| Benchmark | Tests | Example |
|-----------|-------|---------|
| **MMLU** | General knowledge | "What is photosynthesis?" |
| **HumanEval** | Coding ability | "Write function to reverse string" |
| **GSM8K** | Math reasoning | "If John has 5 apples..." |
| **HellaSwag** | Common sense | Complete: "He picked up the phone to..." |
| **TruthfulQA** | Factual accuracy | Questions humans often get wrong |

---

## Tools You Need

### Must Install:
```bash
pip install transformers
pip install torch
pip install sentence-transformers
pip install chromadb
pip install datasets
```

### For APIs (optional):
```bash
pip install anthropic
pip install openai
```

### For Fine-tuning:
```bash
pip install peft
pip install accelerate
```

---

## Learning Order (Recommended)

**Week 1-2**: Transformers basics
- Understand embeddings
- Load pre-trained models
- Extract features

**Week 3-4**: Prompt Engineering
- Practice different techniques
- Test with real APIs
- Document what works

**Week 5-6**: RAG Systems
- Build simple retriever
- Add vector database
- Create end-to-end pipeline

**Week 7-8**: Vector Databases
- ChromaDB basics
- Metadata filtering
- Performance optimization

**Week 9-10**: Fine-tuning
- Understand LoRA math
- Fine-tune small model
- Compare before/after

**Week 11-12**: Evaluation
- Create benchmarks
- Test different approaches
- Measure improvements

---

## Practice Projects

1. **Personal Document Q&A**
   - Upload your notes/PDFs
   - Ask questions
   - Uses: RAG + Vector DB

2. **Code Assistant**
   - Fine-tune on your codebase style
   - Uses: Fine-tuning + Prompt engineering

3. **Semantic Search Engine**
   - Search your emails/documents by meaning
   - Uses: Vector DB + Embeddings

4. **Benchmark Tool**
   - Test different models
   - Compare performance
   - Uses: Evaluation

---

## Common Mistakes to Avoid

❌ **Don't**: Jump straight to fine-tuning
✅ **Do**: Try better prompts first

❌ **Don't**: Use full fine-tuning for everything
✅ **Do**: Use LoRA/PEFT - it's faster and cheaper

❌ **Don't**: Store raw text in databases
✅ **Do**: Use vector databases for semantic search

❌ **Don't**: Trust outputs blindly
✅ **Do**: Measure and evaluate systematically

❌ **Don't**: Use huge models for simple tasks
✅ **Do**: Start small (GPT-2, DistilBERT) for learning

---

## Key Formulas to Remember

**Attention Score**:
```
score = (Q × K^T) / sqrt(d_k)
attention = softmax(score) × V
```

**Cosine Similarity**:
```
similarity = (A · B) / (||A|| × ||B||)
```

**LoRA Update**:
```
W_new = W_frozen + (A × B)
where A: d×r, B: r×k, r << d,k
```

---

## Resources

- **Hugging Face Course**: Free, hands-on
- **Fast.ai**: Practical approach
- **Andrej Karpathy YouTube**: Deep understanding
- **Papers with Code**: Latest research + code

---

## Your Next Step

Pick ONE skill above and spend 3 days on it:
- Day 1: Understand concept
- Day 2: Write code yourself
- Day 3: Build something with it

Then move to next skill.

**Don't rush. Understanding > Speed.**
