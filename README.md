# 🤖 Prompt Chaining Agentic Design – Machine Learning Classifier Agent

This project demonstrates a **Prompt Chaining + Agentic Workflow Design Pattern** using **LangGraph** and **Groq LLMs**.  
The agent autonomously classifies user queries to determine if they are related to **Machine Learning** and then decides the next action — answer or exit — using a **stateful, graph-driven flow**.

---

## 🧠 What Is Prompt Chaining + Agentic Design?

> **Prompt Chaining** – Breaking a complex reasoning task into multiple, smaller prompts that feed into each other.  
> **Agentic Design** – Designing autonomous decision-making nodes (agents) that work together in a graph to achieve an outcome.

In this project:
- The **Classifier Agent** decides if the query is ML-related.  
- The **Answer Agent** responds intelligently if relevant.  
- The **Graph Controller** (LangGraph) manages transitions between these agents.  

---

## 🏗️ Workflow Architecture

```mermaid
graph TD
    A[START] --> B[🧩 Build Groq Model]
    B --> C[🧠 Classify Query (Prompt 1)]
    C -->|✅ ML Topic| D[🤖 Answer Query (Prompt 2)]
    C -->|🚫 Not ML| E[🔚 Exit]
    D --> F[END]
    E --> F
```

Each node (agent) performs a well-defined **prompt-based reasoning task**, and the system decides the next node based on the output — a key feature of **agentic workflow design**.

---

## ⚙️ Technologies Used

| Component | Purpose |
|------------|----------|
| **LangGraph** | Builds the agentic workflow graph |
| **LangChain Core** | Manages prompts and model chaining |
| **Groq LLM** (`openai/gpt-oss-20b`) | High-speed reasoning and response generation |
| **Rich** | Beautiful terminal UI for enhanced UX |
| **Python Dotenv** | Secure API key management |

---

## 🚀 Features

✅ **Prompt Chaining Workflow:** One prompt classifies, another generates the answer  
✅ **Agentic Decision Flow:** Dynamic branching (continue or exit)  
✅ **Groq-Powered Reasoning:** Ultra-fast and efficient LLM responses  
✅ **Rich Terminal UI:** Visually appealing and interactive experience  
✅ **Loop Mode:** Continuous Q&A with graceful exit commands  

---

## ⚡ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ldotmithu/prompt-chaining-agentic-design.git
cd prompt-chaining-agentic-design
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create .env File
```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🧩 Run the Agentic Workflow
```bash
python main.py
```

### Example:
```
✨ Machine Learning Classifier Agent ✨
❓ Ask your question (or type 'exit' to quit'): What is overfitting in machine learning?

🧩 Processing your query...
✅ The query was related to Machine Learning!

🧠 ML Expert Response:
Overfitting occurs when a model fits training data too closely, capturing noise...
```

---

## 🧪 Prompt Chain Overview

| Stage | Agent | Prompt Function | Description |
|--------|--------|----------------|--------------|
| 1️⃣ | **Build Model** | Initializes Groq LLM | Loads API and prepares reasoning model |
| 2️⃣ | **Classifier Agent** | “Is this query about ML?” | Binary classification (True/False) |
| 3️⃣ | **Answer Agent** | “Answer under 200 words.” | Generates ML-specific answer |
| 4️⃣ | **Graph Controller** | Conditional routing | Directs the flow to next agent or END |

---

## 🧩 Project Structure
```
├── main.py                # Core agentic workflow logic
├── requirements.txt        # Dependencies
├── .env                    # API key file
├── README.md               # Documentation
```

---

## 🔮 Future Enhancements
- 🌐 Multi-topic classification (AI, Data Science, Deep Learning, NLP)
- 🧩 Integrate a memory agent for contextual follow-ups
- 💬 Streamlit or FastAPI UI interface
- 🧱 Plug-and-play agent modules (RAG, Code Analysis, etc.)

---

## 👨‍💻 Author
**Mithurshan**  
🎓 Undergraduate | 🧠 Aspiring Data Scientist  
💼 [GitHub](https://github.com/ldotmithu) · 🌐 [LinkedIn](https://www.linkedin.com/in/mithurshan6)