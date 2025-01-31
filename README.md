# Prédiction de Sentiments des Tweets avec MLOps

## 📌 Description du Projet
Ce projet vise à prédire le sentiment associé à un tweet (positif ou négatif) en utilisant différentes approches d'apprentissage automatique et de deep learning. L'API déployée sur Heroku permet d'envoyer un tweet en entrée et de recevoir une prédiction de sentiment en sortie.

## 🚀 Fonctionnalités
- **Trois approches de modélisation** :
  - Modèle classique (régression logistique)
  - Modèle avancé (LSTM avec embeddings)
  - Modèle BERT
- **Gestion des expérimentations avec MLFlow** :
  - Suivi des performances des modèles
  - Stockage et versioning des modèles
- **Déploiement de l'API sur Heroku**
- **Interface de test avec Streamlit et Postman**
- **Suivi de performance avec Azure Application Insights**
  - Logs des prédictions et erreurs
  - Alerte e-mail en cas d'anomalie

## 📂 Arborescence du Repository
```
├── .github/workflows/    # CI/CD (Tests et déploiement automatique)
├── tests/                # Tests unitaires
├── mlruns/               # Logs des expériences MLFlow
├── .heroku-repo/         # Déploiement Heroku
├── app.py                # API FastAPI
├── streamlit_app.py      # Interface utilisateur
├── requirements.txt      # Dépendances Python
├── Procfile              # Configuration Heroku
├── README.md             # Documentation
├── runtime.txt           # Version Python pour Heroku
├── tokenizer.pkl         # Tokenizer sauvegardé
├── lstm_model.keras      # Modèle entraîné (LSTM)
├── mlflow.db             # Base de données MLFlow
```

## 🛠 Installation et Exécution
### 1️⃣ Cloner le Repository
```bash
git clone https://github.com/ton-repo/mlops-tweet-analysis.git
cd mlops-tweet-analysis
```

### 2️⃣ Installer les Dépendances
```bash
pip install -r requirements.txt
```

### 3️⃣ Lancer l'API Localement
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
L'API sera accessible sur : `http://localhost:8000`

### 4️⃣ Tester l'API avec Postman
Envoyer une requête POST avec un tweet :
```json
{
  "tweet": "Le vol a été annulé, je suis très mécontent !"
}
```
Réponse attendue :
```json
{
  "sentiment": "négatif"
}
```

### 5️⃣ Lancer l'Interface Streamlit
```bash
streamlit run streamlit_app.py
```

## 🔗 Liens Utiles
- **API en production** : [Lien Heroku](https://news-app-azure-47b93008885f.herokuapp.com/)
- **MLFlow Tracking** : [Lien MLFlow](http://localhost:5002)
- **Dépôt GitHub** : [Lien vers GitHub](https://github.com/ilb93/test_api.git)
- **Azure Application Insights** : Configuration et logs visibles dans Azure Portal

## ✨ Auteurs
- **Mourad** - Développement et mise en production
- **MIC - Marketing Intelligence Consulting** - Commanditaire du projet

---
🚀 Ce projet illustre une démarche complète d'IA appliquée avec MLOps pour la gestion et le déploiement efficace de modèles de Machine Learning.
