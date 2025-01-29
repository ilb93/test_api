import requests

BASE_URL = "http://127.0.0.1:5000"  # Change cette URL si nécessaire (ex: URL de ton API sur Heroku)

def test_prediction_success():
    """Test de prédiction avec une requête valide"""
    payload = {
        "tweets": ["J'adore ce produit", "C'est une catastrophe"],
        "expected_labels": [1, 0]
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data
    assert len(json_data["predictions"]) == len(payload["tweets"])

def test_prediction_missing_field():
    """Test d'erreur lorsque 'expected_labels' est manquant"""
    payload = {"tweets": ["J'adore ce produit"]}
    response = requests.post(f"{BASE_URL}/predict", json=payload)

    assert response.status_code == 400
    assert "error" in response.json()

def test_prediction_invalid_data():
    """Test d'erreur lorsque les données ne sont pas sous forme de liste"""
    payload = {"tweets": "Ce produit est génial", "expected_labels": [1]}
    response = requests.post(f"{BASE_URL}/predict", json=payload)

    assert response.status_code == 400
    assert "error" in response.json()

def test_prediction_mismatch_length():
    """Test d'erreur lorsque 'tweets' et 'expected_labels' n'ont pas la même longueur"""
    payload = {
        "tweets": ["J'adore ce produit", "C'est une catastrophe"],
        "expected_labels": [1]  # Manque un élément
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)

    assert response.status_code == 400
    assert "error" in response.json()

if __name__ == "__main__":
    print("🔹 Exécution des tests...")
    test_prediction_success()
    test_prediction_missing_field()
    test_prediction_invalid_data()
    test_prediction_mismatch_length()
    print("✅ Tous les tests ont été exécutés avec succès.")
