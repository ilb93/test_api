import streamlit as st
import requests

# Configuration de la page Streamlit
st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="centered")

# Titre de l'application
st.title("Analyse de Sentiment des Tweets 🐦")

# Instructions
st.markdown("""
Entrez un ou plusieurs tweets dans le champ ci-dessous, puis cliquez sur **Analyser** pour voir les sentiments (positif ou négatif).
""")

# Entrée utilisateur : Zone de texte pour saisir les tweets
tweets_input = st.text_area(
    "Entrez vos tweets (un par ligne) :", 
    placeholder="Exemple :\nJ'adore ce produit !\nCe service est horrible...",
    height=200
)

# Bouton pour envoyer les données
if st.button("Analyser"):

    # Vérification des données saisies
    if tweets_input.strip():
        # Préparer les données pour l'API
        tweets_list = [tweet.strip() for tweet in tweets_input.split("\n") if tweet.strip()]
        payload = {"tweets": tweets_list}

        # Appeler l'API
        try:
            # Remplacez l'URL par celle de votre API déployée
            api_url = "https://news-app-azure-47b93008885f.herokuapp.com/predict"
            response = requests.post(api_url, json=payload)

            # Vérifiez si la requête a réussi
            if response.status_code == 200:
                results = response.json().get("predictions", [])
                st.success("Analyse terminée avec succès !")
                
                # Afficher les résultats
                for result in results:
                    tweet = result.get("tweet")
                    sentiment = result.get("sentiment")
                    sentiment_label = "😊 Positif" if sentiment == "positif" else "😡 Négatif"
                    st.write(f"- **Tweet** : {tweet}")
                    st.write(f"  - **Sentiment** : {sentiment_label}")
                    st.write("---")
            else:
                st.error("Erreur lors de la connexion à l'API.")
                st.write(f"Code d'erreur : {response.status_code}")
                st.write(response.json())
        except Exception as e:
            st.error("Une erreur est survenue lors de la connexion à l'API.")
            st.write(f"Détails : {e}")
    else:
        st.warning("Veuillez entrer au moins un tweet.")

# Footer
st.markdown("""
---
Fait avec ❤️ par [Votre Nom]
""")
