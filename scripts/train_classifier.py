#!/usr/bin/env python3
"""
Semantic Overlap Classifier Training
Trains a classifier to detect: duplicate, subsumption, conflict, delegation, different
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"


class OverlapClassifier:
    """Classifier for semantic overlap types"""
    
    OVERLAP_TYPES = [
        'duplicate',      # Nearly identical text
        'subsumption',    # One text contains/implies the other
        'conflict',       # Contradictory provisions
        'delegation',     # One delegates to the other
        'different'       # Semantically different
    ]
    
    def __init__(self):
        self.classifier = None
        self.feature_names = []
    
    def create_training_data(self, similarity_df: pd.DataFrame, corpus_df: pd.DataFrame) -> pd.DataFrame:
        """Create labeled training data from similarity pairs"""
        
        print("\nCreating training data...")
        
        training_data = []
        
        for _, row in similarity_df.iterrows():
            idx1, idx2 = row['idx1'], row['idx2']
            similarity = row['similarity']
            
            # Get text data
            text1 = corpus_df.iloc[idx1]['text']
            text2 = corpus_df.iloc[idx2]['text']
            
            # Extract features
            features = self._extract_features(text1, text2, similarity, row)
            
            # Auto-label based on heuristics
            label = self._auto_label(features, row)
            
            features['label'] = label
            training_data.append(features)
        
        df = pd.DataFrame(training_data)
        
        print(f"✓ Created {len(df)} training samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def _extract_features(self, text1: str, text2: str, similarity: float, row: Dict) -> Dict:
        """Extract features for classification"""
        
        len1 = len(text1)
        len2 = len(text2)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Compute features
        features = {
            'similarity': similarity,
            'len_ratio': min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0,
            'len_diff': abs(len1 - len2),
            'word_overlap': len(words1 & words2) / len(words1 | words2) if len(words1 | words2) > 0 else 0,
            'same_doc': int(row.get('same_doc', False)),
            'cross_group': int(row.get('cross_group', False)),
            'avg_length': (len1 + len2) / 2,
            'max_length': max(len1, len2),
            'min_length': min(len1, len2)
        }
        
        return features
    
    def _auto_label(self, features: Dict, row: Dict) -> str:
        """Auto-label based on heuristics"""
        
        sim = features['similarity']
        len_ratio = features['len_ratio']
        cross_group = features['cross_group']
        
        # Duplicate: very high similarity, similar length
        if sim > 0.95 and len_ratio > 0.8:
            return 'duplicate'
        
        # Subsumption: high similarity but different lengths
        elif sim > 0.85 and len_ratio < 0.7:
            return 'subsumption'
        
        # Delegation: cross-group with high similarity
        elif cross_group and sim > 0.80:
            return 'delegation'
        
        # Different: lower similarity
        elif sim < 0.75:
            return 'different'
        
        # Default: subsumption
        else:
            return 'subsumption'
    
    def train(self, training_df: pd.DataFrame):
        """Train the classifier"""
        
        print("\nTraining classifier...")
        
        # Prepare features and labels
        feature_cols = [col for col in training_df.columns if col != 'label']
        X = training_df[feature_cols].values
        y = training_df['label'].values
        
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train Random Forest
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        
        print("\n" + "=" * 60)
        print("Classification Report:")
        print("=" * 60)
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importances = self.classifier.feature_importances_
        feature_importance = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\nFeature Importance:")
        for feat, imp in feature_importance:
            print(f"  {feat}: {imp:.4f}")
        
        # Compute accuracy
        accuracy = (y_pred == y_test).mean()
        print(f"\nAccuracy: {accuracy:.2%}")
        
        return {
            'accuracy': float(accuracy),
            'feature_importance': {feat: float(imp) for feat, imp in feature_importance}
        }
    
    def predict(self, features: Dict) -> str:
        """Predict overlap type for a pair"""
        
        if self.classifier is None:
            raise ValueError("Classifier not trained yet")
        
        # Extract feature vector
        X = np.array([[features[col] for col in self.feature_names]])
        
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        return prediction, dict(zip(self.classifier.classes_, probabilities))
    
    def save(self, filepath: Path):
        """Save classifier to disk"""
        
        model_data = {
            'classifier': self.classifier,
            'feature_names': self.feature_names,
            'overlap_types': self.OVERLAP_TYPES
        }
        
        joblib.dump(model_data, filepath)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Saved classifier to {filepath} ({size_mb:.2f} MB)")
    
    @classmethod
    def load(cls, filepath: Path):
        """Load classifier from disk"""
        
        model_data = joblib.load(filepath)
        
        instance = cls()
        instance.classifier = model_data['classifier']
        instance.feature_names = model_data['feature_names']
        
        print(f"✓ Loaded classifier from {filepath}")
        
        return instance


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Semantic Overlap Classifier Training")
    print("=" * 60)
    
    # Load similarity pairs
    print("\n[1/4] Loading data...")
    similarity_path = PROCESSED_DIR / "similarity_pairs.parquet"
    corpus_path = PROCESSED_DIR / "lovdata_corpus.parquet"
    
    if not similarity_path.exists():
        print(f"ERROR: Similarity pairs not found at {similarity_path}")
        print("Please run embedding_pipeline.py first")
        sys.exit(1)
    
    similarity_df = pd.read_parquet(similarity_path)
    corpus_df = pd.read_parquet(corpus_path)
    
    print(f"✓ Loaded {len(similarity_df)} similarity pairs")
    print(f"✓ Loaded {len(corpus_df)} corpus texts")
    
    # Create classifier
    print("\n[2/4] Preparing training data...")
    classifier = OverlapClassifier()
    training_df = classifier.create_training_data(similarity_df, corpus_df)
    
    # Save training data
    training_path = PROCESSED_DIR / "overlap_training_data.parquet"
    training_df.to_parquet(training_path, index=False)
    print(f"✓ Saved training data to {training_path}")
    
    # Train classifier
    print("\n[3/4] Training classifier...")
    metrics = classifier.train(training_df)
    
    # Save classifier
    print("\n[4/4] Saving classifier...")
    model_path = MODELS_DIR / "overlap_classifier.joblib"
    classifier.save(model_path)
    
    # Save metrics
    metrics_path = MODELS_DIR / "classifier_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Classifier Training Complete!")
    print("=" * 60)
    print(f"Training samples: {len(training_df)}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
