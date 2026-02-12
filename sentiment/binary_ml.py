import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ml_train import (
    Config as BaseConfig,
    SENTENCE_TRANSFORMER_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    load_data_enhanced,
    get_sentiment_model,
    predict_sentiment_batch,
    detect_sentiment_advanced,
    create_embeddings,
    get_ml_models,
    create_confusion_matrix,
    create_performance_visualization,
    create_results_table,
)


class Config(BaseConfig):
    """
    Configuration for negative-only prediction task.

    We keep the same DATA_DIR as the main ML training (`../dataset`)
    but write results into a dedicated subfolder so it matches the
    existing structure under `sentiment/outputs`.
    """

    CONFUSION_MATRIX_DIR = os.path.join(
        BaseConfig.OUTPUT_DIR, "negative_ml_confusion_matrix"
    )


def main():
    print("=" * 80)
    print("🎯 NEGATIVE-FOCUSED SENTIMENT PREDICTION (10 ML MODELS, 3-CLASS CONFUSION MATRIX)")
    print("=" * 80)

    if not SENTENCE_TRANSFORMER_AVAILABLE:
        print("❌ Sentence Transformers not available. Install: pip install sentence-transformers")
        return

    # Define embeddings (same as main ML script)
    embedding_models = {
        "MPNet": "all-mpnet-base-v2",
        "all-MiniLM-L6": "all-MiniLM-L6-v2",
    }

    device = "cpu"
    print(f"Using device: {device}\n")

    os.makedirs(Config.CONFUSION_MATRIX_DIR, exist_ok=True)
    print(f"📁 Confusion matrices will be saved in: {Config.CONFUSION_MATRIX_DIR}\n")

    # ------------------------------------------------------------------
    # 1) Load full dataset from `dataset/`
    # ------------------------------------------------------------------
    print("📂 Loading and preprocessing data from full dataset...")
    df = load_data_enhanced()
    if df is None:
        return

    # ------------------------------------------------------------------
    # 2) Generate multi-class ground truth with RoBERTa (0/1/2)
    # ------------------------------------------------------------------
    print("🔍 Using RoBERTa as ground truth (Negative / Neutral / Positive)...")

    if not TRANSFORMERS_AVAILABLE:
        print("   ❌ Transformers not available. Install: pip install transformers")
        return

    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print("   📥 Loading RoBERTa for ground truth...")
    roberta_tokenizer, roberta_model = get_sentiment_model(roberta_model_name)

    if roberta_tokenizer and roberta_model:
        texts = df["cleaned_text"].tolist()
        ground_truth_predictions = []

        print("   🔮 Generating ground truth labels with RoBERTa...")
        for i in range(0, len(texts), 8):
            try:
                batch_texts = texts[i : i + 8]
                batch_preds = predict_sentiment_batch(
                    batch_texts,
                    roberta_tokenizer,
                    roberta_model,
                    roberta_model_name,
                    device,
                )
                ground_truth_predictions.extend(batch_preds)
                gc.collect()
                time.sleep(0.01)
            except Exception as e:
                print(f"\n         ⚠️  Error in batch {i // 8 + 1}: {e}")
                ground_truth_predictions.extend([-1] * len(batch_texts))
                gc.collect()
                continue

        while len(ground_truth_predictions) < len(texts):
            ground_truth_predictions.append(-1)

        df["ground_truth_mc"] = ground_truth_predictions

        invalid_mask = df["ground_truth_mc"] == -1
        if invalid_mask.sum() > 0:
            print(
                f"   ⚠️  {invalid_mask.sum()} invalid RoBERTa predictions, using keyword-based fallback..."
            )
            df.loc[invalid_mask, "ground_truth_mc"] = df.loc[
                invalid_mask, "cleaned_text"
            ].apply(detect_sentiment_advanced)

        del roberta_model
        del roberta_tokenizer
        gc.collect()
        time.sleep(0.5)
    else:
        print("   ⚠️  Failed to load RoBERTa, using keyword-based ground truth...")
        df["ground_truth_mc"] = df["cleaned_text"].apply(detect_sentiment_advanced)

    sentiment_counts = df["ground_truth_mc"].value_counts().sort_index()
    sentiment_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print("📊 Multi-class ground truth sentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment_names.get(sentiment, sentiment)}: {count} ({percentage:.1f}%)")

    # Use the full 3-class ground truth (0=Negative,1=Neutral,2=Positive)
    df["ground_truth"] = df["ground_truth_mc"]

    texts = df["cleaned_text"].tolist()
    y_true = df["ground_truth"].tolist()

    print(f"\n📊 Total samples: {len(texts)}")
    class_counts = df["ground_truth"].value_counts().sort_index()
    print("📊 Label distribution (0=Negative, 1=Neutral, 2=Positive):")
    for label, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 3) Train / test split
    # ------------------------------------------------------------------
    if len(set(y_true)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_true, test_size=0.2, random_state=42, stratify=y_true
        )
    else:
        # Edge case: if everything is one class (unlikely, but safe-guard)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_true, test_size=0.2, random_state=42
        )

    print("📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing:  {len(X_test)} samples")

    # ------------------------------------------------------------------
    # 4) Prepare models
    # ------------------------------------------------------------------
    ml_models = get_ml_models()
    print(f"\n🤖 Available ML models: {len(ml_models)}")
    for model_name in ml_models.keys():
        print(f"   - {model_name}")

    all_results = []

    # ------------------------------------------------------------------
    # 5) For each embedding: create embeddings → train 10 models → metrics + confusion
    # ------------------------------------------------------------------
    for emb_name, emb_model_name in embedding_models.items():
        print("\n" + "=" * 70)
        print(f"📦 PROCESSING EMBEDDING: {emb_name.upper()}")
        print("=" * 70)

        print(f"   📥 Creating embeddings with {emb_name}...")
        X_train_emb = create_embeddings(X_train, emb_model_name)
        X_test_emb = create_embeddings(X_test, emb_model_name)

        if X_train_emb is None or X_test_emb is None:
            print(f"   ❌ Failed to create embeddings for {emb_name}")
            continue

        print(
            f"   ✅ Embeddings created: Train shape {X_train_emb.shape}, Test shape {X_test_emb.shape}"
        )

        # Scaling
        scaler_standard = StandardScaler()
        X_train_scaled_standard = scaler_standard.fit_transform(X_train_emb)
        X_test_scaled_standard = scaler_standard.transform(X_test_emb)

        scaler_minmax = MinMaxScaler()
        X_train_scaled_minmax = scaler_minmax.fit_transform(X_train_emb)
        X_test_scaled_minmax = scaler_minmax.transform(X_test_emb)

        for model_name, model in ml_models.items():
            print(f"\n   🤖 Training {model_name}...")

            try:
                if model_name == "Naive Bayes":
                    X_train_use = X_train_scaled_minmax
                    X_test_use = X_test_scaled_minmax
                elif model_name in ["SVM", "Logistic Regression"]:
                    X_train_use = X_train_scaled_standard
                    X_test_use = X_test_scaled_standard
                else:
                    X_train_use = X_train_emb
                    X_test_use = X_test_emb

                # Fit and predict
                model.fit(X_train_use, y_train)
                y_pred = model.predict(X_test_use)

                accuracy = accuracy_score(y_test, y_pred) * 100
                precision = (
                    precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    * 100
                )
                recall = (
                    recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    * 100
                )
                f1 = (
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    * 100
                )

                all_results.append(
                    {
                        "Model": model_name,
                        "Embedding": emb_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                    }
                )

                print(
                    f"      ✅ Accuracy: {accuracy:.2f}%, "
                    f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}%"
                )

                print("      📊 Creating confusion matrix...")
                create_confusion_matrix(
                    y_test,
                    y_pred,
                    model_name,
                    emb_name,
                    Config.CONFUSION_MATRIX_DIR,
                    accuracy=accuracy,
                )

                del model
                gc.collect()
            except Exception as e:
                print(f"      ❌ Error training {model_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Cleanup per-embedding
        del X_train_emb, X_test_emb
        del X_train_scaled_standard, X_test_scaled_standard
        del X_train_scaled_minmax, X_test_scaled_minmax
        gc.collect()
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # 6) Aggregate results → plots + table (same style as existing outputs)
    # ------------------------------------------------------------------
    if all_results:
        print("\n" + "=" * 80)
        print("📊 GENERATING RESULTS (NEGATIVE-ONLY BINARY TASK)")
        print("=" * 80)

        results_df = pd.DataFrame(all_results)

        create_performance_visualization(results_df, Config.CONFUSION_MATRIX_DIR)
        create_results_table(results_df, Config.CONFUSION_MATRIX_DIR)

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE (NEGATIVE-ONLY BINARY TASK)")
        print("=" * 80)
        print(f"📊 Total model-embedding combinations processed: {len(results_df)}")
        print(f"📈 Best accuracy: {results_df['Accuracy'].max():.2f}%")
        best_row = results_df.loc[results_df["Accuracy"].idxmax()]
        print(
            f"🏆 Best combination: {best_row['Model']} with {best_row['Embedding']} "
            f"(Accuracy: {best_row['Accuracy']:.2f}%, F1: {best_row['F1']:.2f}%)"
        )
        print(f"\n📁 Results saved in: {Config.OUTPUT_DIR}")
        print(f"📊 Confusion matrices saved in: {Config.CONFUSION_MATRIX_DIR}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()


