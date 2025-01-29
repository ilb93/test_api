import requests

BASE_URL = "http://127.0.0.1:5000"  # Mets ici l'URL de ton API en production si nécessaire

def test_prediction():
    """Test une requête valide à l'API de prédiction"""
    payload = {"tweets": ["J'adore ce produit"]}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200  # Vérifie que l'API répond bien avec succès
    assert "predictions" in response.json()  # Vérifie que la clé "predictions" est présente

def test_prediction_missing_field():
    """Test d'erreur lorsqu'un champ est manquant"""
    payload = {}  # On ne met pas "tweets" pour forcer une erreur
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 400  # Vérifie que l'API renvoie bien une erreur 400
    assert "error" in response.json()  # Vérifie qu'un message d'erreur est retourné

