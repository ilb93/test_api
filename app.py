from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import logging
from datetime import datetime, timedelta
from opencensus.ext.azure.log_exporter import AzureLogHandler
import os

# ------------------------------
# 🔹 CONFIGURATION DES LOGS AZURE
# ------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    azure_handler = AzureLogHandler(connection_string="InstrumentationKey=47019b65-b8ca-40be-95c8-a0552c3b62b3")
    logger.addHandler(azure_handler)
    logger.info("✅ Azure Application Insights Logging configuré avec succès.")
except Exception as e:
    print(f"⚠️ Erreur lors de la configuration d'Azure Application Insights Logging : {e}")

# ------------------------------
# 🔹 CHARGEMENT DU MODÈLE ET TOKENIZER
# ------------------------------
MODEL_PATH = "lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

if not os.path.exists(MODEL_PATH):
    logger.error("❌ Le fichier lstm_model.keras est introuvable !")
    model = None
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("✅ Modèle 'lstm_model.keras' chargé avec succès.")
    except Exception as e:
        logger.error(f"⚠️ Erreur lors du chargement du modèle : {e}")
        model = None

if not os.path.exists(TOKENIZER_PATH):
    logger.error("❌ Le fichier tokenizer.pkl est introuvable !")
    tokenizer = None
else:
    try:
        tokenizer = joblib.load(TOKENIZER_PATH)
        logger.info("✅ Tokenizer 'tokenizer.pkl' chargé avec succès.")
    except Exception as e:
        logger.error(f"⚠️ Erreur lors du chargement du tokenizer : {e}")
        tokenizer = None

# ------------------------------
# 🔹 INITIALISATION DE L'APPLICATION FLASK
# ------------------------------
app = Flask(__name__)

# Stockage temporaire des erreurs de prédiction
misclassified_predictions = []

@app.route("/")
def home():
    return jsonify({"message": "Bienvenue sur l'API de prédiction de sentiments !"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Vérifier si le modèle et le tokenizer sont chargés
        if model is None or tokenizer is None:
            logger.error("❌ Le modèle ou le tokenizer n'est pas chargé.")
            return jsonify({"error": "Le modèle ou le tokenizer n'est pas chargé."}), 500

        # Vérifier les données d'entrée
        data = request.json
        if "tweets" not in data:
            logger.error("❌ Le champ 'tweets' est requis.")
            return jsonify({"error": "Le champ 'tweets' est requis."}), 400

        tweets = data["tweets"]

        if not isinstance(tweets, list):
            logger.error("❌ Les données fournies ne sont pas valides.")
            return jsonify({"error": "Les données fournies ne sont pas valides."}), 400

        # ------------------------------
        # 🔹 PRÉTRAITEMENT DES TWEETS
        # ------------------------------
        sequences = tokenizer.texts_to_sequences(tweets)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)

        # Prédiction des sentiments
        predictions = (model.predict(padded) > 0.5).astype("int32").flatten()

        results = []
        now = datetime.utcnow()
        detected_errors = []  # Stocke uniquement les erreurs pour ce batch

        # ------------------------------
        # 🔹 TEST DES PRÉDICTIONS
        # ------------------------------
        expected_labels = data.get("expected_labels")  # Peut être None

        for i, (tweet, prediction) in enumerate(zip(tweets, predictions)):
            sentiment = "positif" if prediction == 1 else "négatif"
            result = {"tweet": tweet, "prediction": sentiment}

            if expected_labels and len(expected_labels) == len(tweets):
                expected_sentiment = "positif" if expected_labels[i] == 1 else "négatif"
                result["expected"] = expected_sentiment

                # Vérifier si la prédiction est incorrecte
                if sentiment != expected_sentiment:
                    detected_errors.append({"timestamp": now, "tweet": tweet, "prediction": sentiment, "expected": expected_sentiment})
                    logger.warning(f"⚠️ TWEET MAL PREDIT : tweet='{tweet}', prediction='{sentiment}', attendu='{expected_sentiment}'")

            results.append(result)

        # Ajouter les erreurs détectées à la liste globale
        misclassified_predictions.extend(detected_errors)

        # Supprimer les erreurs plus vieilles que 5 minutes
        misclassified_predictions[:] = [e for e in misclassified_predictions if e["timestamp"] > now - timedelta(minutes=5)]

        # ------------------------------
        # 🔹 DÉCLENCHEMENT DE L'ALERTE AZURE
        # ------------------------------
        if len(misclassified_predictions) >= 3:
            logger.error(f"🚨 ALERTE : 3 prédictions incorrectes détectées en moins de 5 minutes ! Détails : {misclassified_predictions}")

        return jsonify({"predictions": results, "errors_detected": detected_errors})

    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("🚀 Application Flask en cours d'exécution...")
<<<<<<< Updated upstream
    app.run(debug=False, host="0.0.0.0", port=5000)

=======
    app.run(debug=False, host="0.0.0.0", port=5000)
>>>>>>> Stashed changes
