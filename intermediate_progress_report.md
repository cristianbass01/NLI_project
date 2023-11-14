## Intermediate Progress Report: Natural Language Interaction Project

Cristian Bassotto   - u210426  
Nikolay Kormushev   - u234126  
Adrian Gheorgiu     - <TODO>  
Giuseppe Cianci     - u234127


### Task 1: Domain Identification/Dialog Act Prediction

#### Approaches Tried
- **Feature Engineering:** Preprocess data with TF-IDF (Term Frequency – Inverse Document Frequency) and use it to extract more features to train on
- **SVM:** SVM with rbf kernel using fasttext embeddings and lemmatization  
  Accuracy: 76.9 %, Precision: 69.4 %, Recall: 64.1 %, F1: 66.5 %
- **XGBRFClassifier:** Extreme Gradient Boosted Random Forest Classifier with unigram frequency features  
  Accuracy: 63.3 %
- **Finetuning Pre-trained Transformer Models:** Utilized BERT for fine-tuning on our specific dataset.  
  BERT: Accuracy: 86.6 %, Precision: 89.8 %, Recall: 89.2 %, F1: 89.5 %  

#### Possible Improvements
- Experiment with different transformer models (e.g., GPT, RoBERTa) for potentially better contextual understanding.
- Integrate linguistically motivated rule-based post-processing to refine predictions.

### Task 2: Content Extraction from User Utterances (Semantic Frame Slot Filling)

#### Approaches Tried
- **Finetuning Pre-trained Transformer Models:** Utilized BERT for fine-tuning on our specific dataset.  
  Accuracy: 98 %, Precision: 91 %, Recall: 95 %, F1-Score: 93 %

#### Possible Improvements
- The performance is already very good, so we are going to improve the other tasks first.

### Task 3: Agent Move Prediction

Not started yet.