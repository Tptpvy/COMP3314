@echo off

REM Install Python packages
pip install -r requirements.txt

REM Download spaCy models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

REM Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"

echo Setup completed successfully!
pause