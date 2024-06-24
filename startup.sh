#!/bin/sh

# Pull DVC data
dvc pull prod_models.dvc

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
