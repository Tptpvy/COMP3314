#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy models
python3 -m spacy download de_core_news_sm
python3 -m spacy download en_core_web_sm

# Download NLTK data for BLEU score
python3 -c "import nltk; nltk.download('punkt')"

echo "Setup complete!"