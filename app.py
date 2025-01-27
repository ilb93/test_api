from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import logging
from google.cloud import logging as cloud_logging
from datetime import datetime, timedelta

# Configurer Google Cloud Logging
try:
    cloud_client = cloud_logging.Client()
    cloud_client.setup_logging()
    logging.info("Google Cloud Logging configuré avec succès.")
except Exception as e:
    logging.error(f"Erreur lors de la configuration de Google Cloud Logging : {e}")

# Charger le modèle et le tokenizer
try:
    model = tf.keras.models.load_model("model_lstm_v2.keras")
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    model = None

try:
    tokenizer = joblib.load("tokenizer.pkl")
    logging.info("Tokenizer chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du tokenizer : {e}")
    tokenizer = None

# Initialiser l'application Flask
app = Flask(__name__)

# Stockage des tweets négatifs et leurs timestamps
negative_tweets = []

@app.route("/")
def home():
    return jsonify({"message": "Bienvenue sur l'API de prédiction de sentiments !"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Vérifier si le modèle et le tokenizer sont chargés
        if model is None or tokenizer is None:
            return jsonify({"error": "Le modèle ou le tokenizer n'est pas correctement chargé."}), 500

        # Récupérer les tweets à analyser
        data = request.json
        if "tweets" not in data:
            return jsonify({"error": "Le champ 'tweets' est manquant"}), 400

        tweets = data["tweets"]
        if not isinstance(tweets, list):
            return jsonify({"error": "Le champ 'tweets' doit être une liste."}), 400

        # Prétraitement des tweets
        sequences = tokenizer.texts_to_sequences(tweets)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)

        # Prédiction des sentiments
        predictions = (model.predict(padded) > 0.5).astype("int32")
        results = []
        now = datetime.utcnow()

        for tweet, pred in zip(tweets, predictions):
            sentiment = "positif" if pred[0] == 1 else "négatif"
            results.append({"tweet": tweet, "sentiment": sentiment})

            # Si le tweet est négatif, l'ajouter à la liste
            if sentiment == "négatif":
                negative_tweets.append({"timestamp": now, "tweet": tweet})
                logging.info(f"Tweet négatif détecté : {tweet}")

        # Supprimer les anciens tweets négatifs (plus de 5 minutes)
        negative_tweets[:] = [t for t in negative_tweets if t["timestamp"] > now - timedelta(minutes=5)]

        # Si plus de 3 tweets négatifs, loguer une alerte
        if len(negative_tweets) > 3:
            logging.warning(f"ALERTE : Plus de 3 tweets négatifs détectés en 5 minutes. Détails : {negative_tweets}")

        return jsonify({"predictions": results})

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
