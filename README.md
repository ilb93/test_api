# 📊 Prédiction de Sentiments des Tweets avec MLOps

## 📌 Description du Projet  
Ce projet vise à **prédire le sentiment d'un tweet** (positif ou négatif) en utilisant différentes approches d'apprentissage automatique et de deep learning.  

L'API déployée sur **Heroku** permet d'envoyer un tweet en entrée et de recevoir une prédiction de sentiment en sortie.  
Le suivi des performances et le monitoring sont effectués grâce à **MLFlow** et **Azure Application Insights**.....

---

## 🧠 Approches de Modélisation  

Trois approches ont été testées :  
1️⃣ **Modèle Classique** : Régression Logistique sur des TF-IDF.  
2️⃣ **Modèle LSTM** : Réseau de neurones récurrent avec embeddings GloVe.  
3️⃣ **Modèle BERT** : Fine-tuning de `DistilBERT` pour classification binaire.

✔️ **Choix final** : **LSTM**, car il offre un bon compromis entre **performance** et **spécificité**.  
   - **Spécificité élevée** sur les tweets négatifs, réduisant les faux positifs.  
   - Temps d’inférence **plus rapide** que BERT.

---

## 🔧 Fonctionnalités  
✅ **API REST** sur Heroku (prévision en temps réel).  
✅ **Interface de test** Streamlit pour visualiser les prédictions.  
✅ **Gestion des expériences** avec **MLFlow** (tracking des modèles).  
✅ **Monitoring en production** via **Azure Application Insights**.  
✅ **Alertes e-mails** en cas d’anomalie (mauvaises prédictions répétées).  
✅ **Tests unitaires** pour garantir le bon fonctionnement de l’API.

---

## 📂 Arborescence du Repository  
📦 projet_sentiment_analysis ├── .github/workflows/ # CI/CD (Tests et déploiement automatique) ├── tests/ # Tests unitaires ├── mlruns/ # Logs des expériences MLFlow ├── .heroku-repo/ # Déploiement Heroku ├── notebooks/ # Notebooks d'entraînement des modèles ├── app.py # API FastAPI ├── streamlit_app.py # Interface utilisateur ├── requirements.txt # Dépendances Python ├── Procfile # Configuration Heroku ├── README.md # Documentation ├── runtime.txt # Version Python pour Heroku ├── tokenizer.pkl # Tokenizer sauvegardé ├── lstm_model.keras # Modèle entraîné (LSTM) ├── mlflow.db # Base de données MLFlow

yaml
Copier
Modifier

---

## 🛠 Installation et Exécution  

### 1️⃣ **Cloner le Repository**  
```bash
git clone https://github.com/ilb93/test_api.git
cd test_api
2️⃣ Installer les Dépendances
bash
Copier
Modifier
pip install -r requirements.txt
3️⃣ Lancer l'API Localement
bash
Copier
Modifier
uvicorn app:app --host 0.0.0.0 --port 8000
L'API sera accessible sur : http://localhost:8000

4️⃣ Tester l'API avec Postman
📩 Requête POST /predict

json
Copier
Modifier
{
  "tweet": "Le vol a été annulé, je suis très mécontent !"
}
🔎 Réponse attendue :

json
Copier
Modifier
{
  "sentiment": "négatif"
}
5️⃣ Lancer l'Interface Streamlit
bash
Copier
Modifier
streamlit run streamlit_app.py
📊 Suivi des Expériences avec MLFlow
📌 MLFlow permet de suivre les modèles et leur performance.
➡️ Accéder à l’interface MLFlow en local :

bash
Copier
Modifier
mlflow ui
➡️ Consulter les logs des expérimentations :
1️⃣ Accéder au dossier mlruns/
2️⃣ Voir les courbes d'entraînement des modèles

🛡️ Tests et Déploiement
✅ Tests unitaires avec pytest

bash
Copier
Modifier
pytest tests/test_api.py
➡️ Vérifie que l'API retourne bien les bons résultats.

✅ CI/CD avec GitHub Actions

Chaque push déclenche un test automatique.
Si les tests passent, l’API est déployée sur Heroku.
📡 Monitoring avec Azure Application Insights
➡️ Envoi de logs sur Azure pour suivre l'API en production.
➡️ Déclenchement d'une alerte email si plus de 3 prédictions incorrectes en 3 minutes.

🔗 Liens Utiles
🚀 API en production : Lien Heroku
📊 MLFlow Tracking : Lien MLFlow
📁 Dépôt GitHub : Lien vers GitHub
📡 Azure Application Insights : Logs et alertes en temps réel.

✨ Auteurs
👨‍💻 Mourad - Développement et mise en production
🏢 MIC - Marketing Intelligence Consulting - Commanditaire du projet

yaml
Copier
Modifier

---

## **📌 Pourquoi cette version est améliorée ?**
✔ **Ajout d'explications** sur **les modèles** et **le choix du LSTM**.  
✔ **Explication détaillée** sur **MLFlow** et **Azure**.  
✔ **Ajout des tests unitaires** et **CI/CD avec GitHub Actions**.  
✔ **Meilleure lisibilité et structuration claire** pour faciliter la compréhension.  
