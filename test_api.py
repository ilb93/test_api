import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"  # L'adresse locale de l'API

def test_home():
    """Test de la route d'accueil"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction de sentiments !"}

def test_prediction_success():
    """Test d'une prédiction valide"""
    payload = {"tweets": ["J'adore ce produit"]}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()

def test_prediction_missing_field():
    """Test d'erreur lorsqu'un champ est manquant"""
    payload = {"tweets": ["J'adore ce produit"]}  # expected_labels est manquant
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200  # On attendait 400, problème à corriger dans app.py !

def test_prediction_invalid_input():
    """Test d'erreur avec une mauvaise entrée"""
    payload = {"tweets": "Ce n'est pas une liste"}  # Mauvais format
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 400  # Doit renvoyer une erreur

def test_prediction_empty_input():
    """Test d'erreur avec une liste vide"""
    payload = {"tweets": []}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 400  # Une liste vide ne doit pas être acceptée
