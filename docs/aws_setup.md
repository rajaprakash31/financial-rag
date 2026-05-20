# Deploying `financial-rag` on AWS

This guide explains how to deploy the `financial-rag` local RAG framework on an AWS EC2 instance.

## 1. Choose the right instance

### Recommended minimum
- `t3.large` or `t4g.large`
- 8–16 GB RAM
- 100 GB EBS storage

### If you need a small local LLM
- `c6i.xlarge` or `g4dn.xlarge`
- 16–32 GB RAM
- 200 GB+ storage

### Notes
- Use `t4g` for Arm64-compatible models and better cost efficiency.
- Use `g4dn` / `g5` only if you plan to use GPU-accelerated local LLMs.

## 2. Provision AWS resources

### EC2
- OS: Ubuntu 22.04 LTS
- Storage: 100+ GB GP3 EBS
- Networking: open SSH only to your IP
- Optional: open HTTP/HTTPS if you expose a service

### Key pair
- Create or use an existing EC2 key pair.
- Store the `.pem` file securely.

## 3. Connect to the instance

```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
```

## 4. Install system dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential curl
```

## 5. Clone the repository

```bash
cd ~
git clone <your-repo-url> financial-rag
cd financial-rag
```

## 6. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Prepare working folders

```bash
mkdir -p data indexes models
```

## 8. Download or copy your models

### Embedding models
- SentenceTransformers models are downloaded automatically on first run.

### Local LLM models (optional)
- Put `.gguf` / `.bin` model files in `~/models/`.
- Example:
  ```bash
  mkdir -p ~/models
  cd ~/models
  curl -L -o gpt4all-lora-quantized.bin \
    https://huggingface.co/nomic-ai/gpt4all-llora/resolve/main/gpt4all-lora-quantized.bin
  ```

## 9. Run the pipeline

### Build the first index

```bash
.venv/bin/python build_index.py --data-dir data --index-root indexes --index-name default --backend faiss
```

### Incrementally update the index

```bash
.venv/bin/python update_index.py --data-dir new_data --index-root indexes --index-name default --backend faiss
```

### Query a named index

```bash
.venv/bin/python query_rag.py --query "What is equity?" --index-root indexes --index-name default
```

### Use the orchestrator

```bash
.venv/bin/python orchestrator.py --query "What is equity?" --index-root indexes
```

### Query with LLM

```bash
.venv/bin/python query_rag.py --query "What is equity?" --index-root indexes --index-name default --llm-model ~/models/gpt4all-lora-quantized.bin
```

## 10. Persist storage

- Keep `indexes/`, `data/`, and `models/` on the EBS volume.
- If using instance stop/start, make sure the same volume is attached.

## 11. Run as a service (optional)

Use `tmux` or `screen` to keep long-running commands alive.

Example with `tmux`:

```bash
sudo apt install -y tmux
tmux new -s financial-rag
source .venv/bin/activate
python orchestrator.py --query "What is equity?" --index-root indexes
```

## 12. Optional production improvements

- Add a FastAPI wrapper for `orchestrator.py`
- Use `systemd` to run a query service
- Add CloudWatch or logging
- Use a larger instance for bigger indexes or LLMs

## 13. Security and cleanup

- Close SSH access when not needed
- Use a security group that restricts access
- Delete the instance when not in use
