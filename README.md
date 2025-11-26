# pyannote-whisper

Run ASR and speaker diarization based on whisper and pyannote.audio.

## Installation
1. Create virtual environment
```bash
python3 -m venv paenv
source paenv/bin/activate  # Linux/macOS
paenv\Scripts\activate     # Windows
```

2. Install whisper 
```bash
pip install openai-whisper
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create Hugging Face Token:
Go To https://huggingface.co/settings/tokens
Create New Token with Read permissions

5. Insert Token into main.py line 47