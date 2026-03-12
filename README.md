# gofundme-animal-campaign-insights

Analysis of GoFundMe Animal-category campaigns to understand how image content, text narratives, and campaign duration relate to fundraising success.

## Business Value
- Helps animal-related nonprofits and individual fundraisers design higher-performing campaigns by identifying what drives engagement and donations.
- Quantifies how visual cues, narrative quality, and timing contribute to outcomes, enabling data-informed messaging templates.
- Provides actionable recommendations (story specificity, visual relevance, urgency framing) that can be operationalized in campaign guidance and review workflows.

## Technical Implementation
- Data collection: Web scraping of GoFundMe Animal-category campaigns (metadata, descriptions, images).
- Image understanding: Google Vision API used to extract top image labels (10 per image).
- Text features: SpaCy 300-dim embeddings for campaign descriptions.
- Modeling: Logistic regression (balanced class weights, 80/20 split). Features evaluated:
  - Image labels + duration
  - Description embeddings + duration
  - Combined image labels (TF-IDF) + description embeddings + log-transformed duration
- Topic modeling: LDA on combined text (image labels + descriptions), coherence-tuned; K=4 selected.

## Key Results
- Best model (combined features) achieved ~64.6% test accuracy, outperforming image-only and text-only baselines.
- LDA topics highlight distinct themes; medical/urgent narratives correlate with higher fundraising, while generic appeals underperform.

## Repository Contents
- `UDA_HW3_Team3.ipynb`: Full analysis notebook (scraping, preprocessing, modeling, topic analysis, recommendations).

## How to Run
1. Open the notebook in Jupyter.
2. Install required dependencies listed in the notebook imports (e.g., `pandas`, `scikit-learn`, `spacy`, `selenium`, `google-cloud-vision`, `gensim`, `pyLDAvis`).
3. Execute cells in order.

## Notes
- Google Vision API requires credentials; set environment variables per Google Cloud instructions before running.
- Web scraping may require a webdriver (e.g., ChromeDriver) installed and on PATH.
