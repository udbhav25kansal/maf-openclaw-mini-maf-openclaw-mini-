# MAF-Openclaw-Mini

A Slack AI Assistant built with **Microsoft Agent Framework** - a Python rewrite of the original TypeScript Openclaw-mini project.

## Project Structure

```
maf-openclaw-mini/
├── venv/                  # Python virtual environment
├── src/                   # Source code
│   ├── agents/            # Agent definitions
│   ├── tools/             # Tool implementations
│   └── ...
├── docs/                  # Documentation
│   └── explanation.md     # Detailed explanations
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Python 3.12+
- Slack Bot Token
- OpenAI API Key (or other LLM provider)

## Setup

1. **Activate virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate

   # Mac/Linux
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

4. **Run the bot:**
   ```bash
   python src/main.py
   ```

## Reference Project

This project is inspired by [Openclaw-mini](../Openclaw-mini), a TypeScript Slack AI assistant with:
- RAG (Retrieval Augmented Generation)
- MCP (Model Context Protocol) integrations
- Long-term memory via mem0
- Task scheduling

## Documentation

See [docs/explanation.md](docs/explanation.md) for detailed explanations of:
- Why we use virtual environments
- Microsoft Agent Framework concepts
- Architecture decisions
- Original project analysis

## License

MIT
