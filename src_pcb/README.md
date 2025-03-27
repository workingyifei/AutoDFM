# PCB DFM Analyzer

A tool for analyzing PCB designs for manufacturability issues using rule-based and LLM-powered approaches.

## Features

- Parse Gerber and Drill files to extract PCB features
- Analyze PCB designs against DFM rules
- Generate DFM reports with violations and recommendations
- Support for supplier-specific capabilities
- FastAPI-based REST API for integration
- Multiple analysis modes (Rule-based, OpenAI, Hugging Face)

## Analysis Modes

The PCB DFM Analyzer supports three analysis modes, providing flexibility based on your needs:

### 1. Rule-Based Analysis (No API keys required)
- Uses basic manufacturing rules defined in YAML files
- Checks PCB features against minimum requirements
- Located in `src_pcb/knowledge_base/dfm_rules/basic_rules.yaml`
- Suitable for basic DFM checks without LLM integration

### 2. OpenAI-Powered Analysis (Requires API Key)
- Enhanced analysis using OpenAI's language models
- More detailed recommendations and contextual insights
- Requires OpenAI API key configuration

### 3. Hugging Face-Powered Analysis (Free option)
- Alternative LLM-based analysis using Hugging Face models
- Uses Mistral-7B-Instruct model by default
- Free to use with Hugging Face API token

## LLM Integration Options

### Option 1: OpenAI (Requires API Key)

1. Get an OpenAI API key from https://platform.openai.com/
2. Set the environment variable:
   ```
   export OPENAI_API_KEY="your-api-key"
   ```
   Or add it to `src_pcb/tokens.json`:
   ```json
   {
     "OPENAI_API_KEY": "your-api-key"
   }
   ```

### Option 2: Hugging Face (Free option)

1. Install the necessary packages:
   ```
   pip install langchain-community huggingface_hub
   ```

2. Get a Hugging Face token from https://huggingface.co/settings/tokens
3. Set the environment variable:
   ```
   export HUGGINGFACE_API_TOKEN="your-token"
   ```
   Or add it to `src_pcb/tokens.json`:
   ```json
   {
     "HUGGINGFACE_API_TOKEN": "your-token"
   }
   ```

### Option 3: Rule-Based Only (No LLM)

If no LLM is configured, the system will automatically fall back to basic rule-based analysis that doesn't require any API keys.

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment variables or create a `src_pcb/tokens.json` file for your preferred LLM provider

## Running the API

```
python -m src_pcb.main
```

The API will be available at http://localhost:8000

## Running the Test Script

Test with different analysis modes:

```bash
# Rule-based analysis
python -m src_pcb.test.test_analyzer --analysis-mode rule-based

# OpenAI-powered analysis
python -m src_pcb.test.test_analyzer --analysis-mode openai

# Hugging Face-powered analysis
python -m src_pcb.test.test_analyzer --analysis-mode huggingface
```

## API Usage

```python
import requests

files = [
    ('gerber_files', open('path/to/top_copper.gbr', 'rb')),
    ('gerber_files', open('path/to/bottom_copper.gbr', 'rb')),
    # Add other Gerber and Drill files
]

response = requests.post(
    'http://localhost:8000/analyze',
    files=files,
    data={
        'supplier_id': 'PCBA_MFG_001',  # Optional
        'analysis_mode': 'huggingface'   # Optional: 'rule-based', 'openai', 'huggingface', or 'auto'
    }
)

dfm_report = response.json()
print(f"Manufacturability Score: {dfm_report['overall_score']}")
print(f"Is Manufacturable: {dfm_report['is_manufacturable']}")
```

## Output Format

The analysis produces a JSON report with the following structure:

```json
{
  "dfm_report": {
    "violations": [
      {
        "rule_id": "string",
        "severity": "string",
        "message": "string",
        "location": null,
        "recommendation": "string"
      }
    ],
    "supplier_id": null,
    "overall_score": 0.95,
    "recommendations": ["string"],
    "is_manufacturable": true,
    "source": "huggingface|openai|rule-based"
  },
  "raw_llm_output": "LLM analysis was performed but output not captured"
}
```

## Troubleshooting

If you encounter an error related to the LLM analysis:
- Verify your API tokens are correctly set in environment variables or tokens.json
- Ensure the LLM models are accessible (internet connection required)
- Check if dependencies are properly installed
- Try falling back to rule-based mode if LLM services are unavailable
