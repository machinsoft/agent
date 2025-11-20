<img width="1024" height="1536" alt="01" src="https://github.com/user-attachments/assets/3032cb55-f086-4fae-bdc5-826a9a7ebb2c" />

# Jinx — Autonomous Engineering Agent

I’m **Jinx** — an autonomous engineering agent built for teams that ship. I turn intent into execution: understand goals, generate code, validate, sandbox, and deliver — all auditable and reproducible by design.

> Enterprise-grade. Minimal surface area. Maximum signal.

---

## 🚀 Features

* **Autonomous loop** — understand → generate → verify → execute → refine.
* **Sandboxed runtime** — isolated async process for secure code execution.
* **Durable memory** — persistent `<evergreen>` store + rolling context compression.
* **Semantic embeddings** — retrieve relevant dialogue or code context.
* **Cognitive core (Brain)** — concept tracking, framework detection, adaptive reasoning.
* **Structured logging** — full trace of model inputs, outputs, and execution results.
* **Micro‑modular architecture** — lightweight, extendable, fault‑tolerant.

> Designed for reliability. Built for regulated and production‑grade environments.

---

## 🔧 Environment Setup

### Python Virtual Environment
Before setting up the project, it's recommended to create a virtual environment. Follow these steps:

Learn about virtual environments: [Python Packaging Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

Before running Jinx, create a virtual environment:

**Windows:**

```
py -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
```

### Project Setup
- Provide an OpenAI API key and configuration via `.env` at project root. See `.env.example` for all keys:

Required:
```
OPENAI_API_KEY=
```

Optional (defaults in code / example):
```
PULSE=120           # initial error-tolerance pulse
TIMEOUT=300         # seconds before autonomous thinking
OPENAI_MODEL=gpt-5  # model override; service falls back to gpt-5 if unset
# PROXY=socks5://127.0.0.1:12334
```

Create `.env` from the example:

Windows (PowerShell):
```
Copy-Item .env.example .env
```

macOS/Linux:
```
cp .env.example .env
```

## 🧠 Quick Start

From a local clone:

```bash
python jinx.py
```

This launches an interactive REPL. Describe a goal — Jinx plans, writes code, tests it in sandbox, and returns results.

---

## 📄 License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 💬 Support

* File issues and feature requests in [GitHub Issues](https://github.com/machinegpt/agent/issues)
* Start a Discussion for architectural or design topics.

---

**Jinx — a system learning to build and evolve itself.**

---

## Experimental Disclaimer

This project is largely written and maintained by an artificial intelligence system. Generated code and outputs may be incomplete, insecure, or incorrect and are not independently verified or audited. Treat everything as experimental software and review/validate all changes before using them in production. Use at your own risk.
