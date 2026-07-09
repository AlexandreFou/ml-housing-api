# ML Housing API

API de prédiction du prix des maisons basée sur un modèle de Machine Learning.

Ce projet met en place un pipeline ML complet avec :

* stockage des données dans PostgreSQL ;
* ingestion des données via un service Docker dédié ;
* entraînement d'un modèle de régression ;
* sauvegarde du modèle entraîné ;
* exposition des prédictions via une API FastAPI ;
* orchestration complète avec Docker Compose.

---

# Architecture

```text
                    Docker Compose

                         |
        -------------------------------------
        |                 |                 |
        v                 v                 v

   PostgreSQL          Trainer           FastAPI
        |                 |                 |
        |                 |                 |
        v                 v                 v

  housing table ---> model.pkl ---> /predict endpoint
```

---

# Technologies utilisées

## Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy

## Backend

* FastAPI
* Uvicorn

## Database

* PostgreSQL
* SQLAlchemy

## Infrastructure

* Docker
* Docker Compose

---

# Installation

## Prérequis

Installer :

* Docker Desktop
* Git

---

## Cloner le projet

```bash
git clone https://github.com/AlexandreFou/ml-housing-api.git

cd ml-housing-api
```

---

# Configuration

Créer un fichier `.env` à la racine du projet :

```env
POSTGRES_USER=mluser
POSTGRES_PASSWORD=mlpassword
POSTGRES_DB=housing
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
```

Ce fichier contient les paramètres de connexion à PostgreSQL.

⚠️ Le fichier `.env` ne doit pas être envoyé sur GitHub.

---

# Lancement du projet

Construire les images Docker :

```bash
docker compose build
```

Démarrer les services :

```bash
docker compose up -d
```

Services disponibles :

* PostgreSQL
* FastAPI

---

# Chargement des données

Le dataset California Housing est chargé dans PostgreSQL grâce au service `loader`.

Commande :

```bash
docker compose run loader
```

Les données sont stockées dans la table :

```text
housing
```

---

# Entraînement du modèle

Lancer l'entraînement :

```bash
docker compose run trainer
```

Le pipeline effectue :

```text
PostgreSQL
      |
      v
Pandas DataFrame
      |
      v
Scikit-learn
      |
      v
model.pkl
```

Le modèle entraîné est sauvegardé dans :

```text
models/model.pkl
```

---

# API de prédiction

La documentation interactive FastAPI est disponible à :

```text
http://localhost:8000/docs
```

Endpoint principal :

```text
POST /predict
```

---

# Exemple de requête

```json
{
  "MedInc": 8.3,
  "HouseAge": 41,
  "AveRooms": 6.9,
  "AveBedrms": 1.0,
  "Population": 322,
  "AveOccup": 2.5,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

Exemple de réponse :

```json
{
  "prediction": 4.52
}
```

---

# Pipeline Machine Learning complet

```text
Dataset California Housing

          |
          v

     PostgreSQL

          |
          v

      SQLAlchemy

          |
          v

       Pandas

          |
          v

   Scikit-learn Model

          |
          v

      model.pkl

          |
          v

      FastAPI

          |
          v

     Prediction API
```

---

# Structure du projet

```text
ml-housing-api/

├── app/
│   ├── api.py
│   ├── train.py
│   ├── predict.py
│   └── load_data.py
│
├── models/
│   └── model.pkl
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

# Commandes utiles

Voir les conteneurs actifs :

```bash
docker ps
```

Voir les logs :

```bash
docker compose logs -f
```

Arrêter les services :

```bash
docker compose down
```

---

# Résultats actuels

Modèle utilisé :

```text
Linear Regression
```

Dataset :

```text
California Housing Dataset
```

Nombre d'observations :

```text
20640
```

Métrique obtenue :

```text
RMSE ≈ 0.75
```

---

# Améliorations futures

* [ ] Ajouter des tests avec Pytest
* [ ] Ajouter du logging professionnel
* [ ] Ajouter GitHub Actions CI/CD
* [ ] Ajouter MLflow pour le suivi des expériences
* [ ] Versionner les modèles
* [ ] Ajouter une validation automatique des données
* [ ] Déployer l'application dans le cloud

---

# Objectif du projet

Ce projet a pour objectif de mettre en pratique les compétences nécessaires pour un rôle de ML Engineer :

* développement Python ;
* manipulation de données SQL ;
* entraînement de modèles ML ;
* création d'API ;
* conteneurisation Docker ;
* premières pratiques MLOps.
