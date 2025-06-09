```markdown
# ⚖️ AutoLawyer – Multi-Agent AI for Legal Case Drafting

AutoLawyer is a full-stack, offline-first LegalTech platform that uses **multi-agent AI systems** powered by **GPT4All (local LLMs)** to automate the drafting of legal briefs. Designed for legal professionals, students, and researchers, it simulates a legal team by coordinating **research**, **drafting**, and **summarization** agents in a LangGraph workflow.

> 🔐 100% Local. No OpenAI or external APIs. Secure. Private. Cost-free AI.

---

## 🧠 Multi-Agent Workflow

| Agent       | Role                                                                 |
|-------------|----------------------------------------------------------------------|
| 🧠 Researcher   | Gathers legal precedents, statutes, and relevant case law       |
| ✍️ Drafter     | Converts research into formal legal arguments                    |
| 📄 Summarizer  | Summarizes the draft into bullet points or executive summary    |

---

## 🛠 Tech Stack

### 🧠 AI
- **LLM**: `GPT4All` using [`Nous Hermes 2 Mistral 7B - Q4_0` GGUF model](https://huggingface.co/TheBloke/Nous-Hermes-2-Mistral-7B-DPO-GGUF)
- **LangGraph**: Defines the AI agent flow

### ⚙ Backend
- **FastAPI**: REST API and agent controller
- **SQLAlchemy**: ORM for DB models
- **Pydantic**: Data validation

### 🖥 Frontend
- **Next.js (TypeScript)**: Web interface
- **Tailwind CSS**: Styling
- **React Quill or TipTap**: Collaborative legal editor

### 🗄 Database
- SQLite (default) or PostgreSQL for production use

---

## 📂 Project Structure

```

autolawyer/
├── backend/
│   ├── agents/             # researcher.py, drafter.py, summarizer.py
│   ├── core/               # LangGraph DAG, prompt templates, runner
│   ├── local\_llm/          # GPT4All runner + config
│   ├── db/                 # SQLAlchemy models and DB setup
│   ├── routes/             # API endpoints (briefs, agents, users)
│   └── main.py             # FastAPI app entrypoint
│
├── frontend/
│   ├── components/         # Editor, AgentView, BriefList
│   ├── pages/              # index.tsx, brief/\[id].tsx
│   └── utils/              # api.ts, formatter.ts
│
├── models/                 # Downloaded GGUF model goes here
├── .env.local              # Local configuration
├── README.md               # This file
└── run\_backend.sh          # Start script for backend and model

````

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/yourusername/AutoLawyer.git
cd AutoLawyer
````

### 2️⃣ Download the LLM Model

Download this model file:

> `Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf`

Place it in:

```
models/Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf
```

### 3️⃣ Backend Setup (Python 3.10+)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 4️⃣ Frontend Setup (Node.js 18+)

```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 Features

* 🧠 Local GPT4All-powered legal brief generation
* 🧩 Modular LangGraph agent design
* 📄 Rich-text legal editor with agent insertions
* 📜 Version-controlled brief logs + full history
* 🛠 Easy agent customization (just swap the file!)
* 🔐 100% offline: No API keys, no cloud

---

## ⚙️ Model Config (backend/local\_llm/model\_config.json)

```json
{
  "model_path": "models/Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
  "context_size": 4096,
  "temperature": 0.7
}
```

---

## 🧠 Example Use Case

1. Create a new brief
2. Trigger the **researcher agent**
3. Review the extracted legal content
4. Trigger the **drafter** to generate arguments
5. Finalize using the **summarizer**
6. Edit, save, or export your draft

---

## 🛡 License

MIT License. No AI-generated content should be considered legal advice.

---

