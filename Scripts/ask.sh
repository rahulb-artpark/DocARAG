#!/usr/bin/env bash
# Ask a question against the already-built BookIndex for the mytest PDF.
# Usage:  bash Scripts/ask.sh "your question here"
set -euo pipefail
cd "$(dirname "$0")/.."

QUESTION="${*:-}"
if [[ -z "$QUESTION" ]]; then
  echo "Usage: bash Scripts/ask.sh \"your question\"" >&2
  exit 1
fi

QA_FILE="datasets/mytest.json"
UUID="47e6394c-4196-5105-8ff0-2b1885bceb3f"
PDF="/home/rahul/data/Aakanksha_Shripal_Master's_Thesis.pdf"

# 1. Write the question into the dataset JSON (keeps it valid; preserves doc ids)
QUESTION="$QUESTION" UUID="$UUID" PDF="$PDF" python3 - "$QA_FILE" <<'PY'
import json, os, sys
path = sys.argv[1]
data = [{
    "question": os.environ["QUESTION"],
    "answer": "",
    "doc_uuid": os.environ["UUID"],
    "doc_path": os.environ["PDF"],
}]
with open(path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Question set to: {os.environ['QUESTION']}")
PY

# 2. Run RAG (index already built; just retrieve + generate)
echo "Running RAG..."
conda run -n gbc-rag python main.py \
  -c config/ollama-gbc.yaml -d Scripts/cfg/mytest.yaml \
  --num 1 --nsplit 1 rag > Scripts/logs/mytest/rag.log 2>&1

# 3. Print the answer
echo "==================== ANSWER ===================="
python3 -c "import json; print(json.load(open('runs/mytest/$UUID/eval_mytest_gbc_standard/final_results.json'))[0]['output'])"
