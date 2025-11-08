#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

# Download NLTK data for BLEU score
python -c "import nltk; nltk.download('punkt')"

echo "Setup complete!"