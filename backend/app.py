from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from prophet import Prophet
from pulp import LpMaximize, LpProblem, LpVariable
import networkx as nx
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()

# Dummy data for visualization
df_demand = pd.DataFrame({
    "ds": pd.date_range(start="2023-01-01", periods=365, freq="D"),
    "y": [50 + i*0.1 + (5 if i%30 == 0 else 0) for i in range(365)]
})

df_inventory = pd.DataFrame({
    "product": ["A", "B"],
    "stock": [200, 300],
    "price": [10, 15]
})

df_routes = nx.Graph()
df_routes.add_weighted_edges_from([
    ("A", "B", 10), ("A", "C", 20), ("B", "C", 5),
    ("B", "D", 15), ("C", "D", 10), ("D", "A", 25)
])

@app.get("/projects")
def get_projects():
    return {
        "title": "Proyectos Machine Learning",
        "projects": [
            "Predicción de Demanda",
            "Optimización de Inventario",
            "Optimización de Rutas",
            "Detección de Anomalías en la Cadena de Suministro"
        ]
    }

@app.get("/predict_demand")
def predict_demand():
    model = Prophet()
    model.fit(df_demand)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].to_dict(orient="records")

@app.get("/optimize_inventory")
def optimize_inventory():
    x = LpVariable("Stock_A", lowBound=0, cat="Integer")
    y = LpVariable("Stock_B", lowBound=0, cat="Integer")
    model = LpProblem(name="Optimización_Stock", sense=LpMaximize)
    model += (x + y <= 500, "Capacidad_almacen")
    model += (10*x + 15*y <= 4000, "Presupuesto")
    model += x + y
    model.solve()
    return {"Stock_A": x.varValue, "Stock_B": y.varValue}

@app.get("/shortest_route")
def shortest_route():
    path = nx.shortest_path(df_routes, source="A", target="D", weight="weight")
    return {"ruta_optima": path}

@app.get("/detect_anomalies")
def detect_anomalies():
    data = np.random.normal(loc=5, scale=1, size=100).tolist()
    data.extend([15, 18, 20])
    df = pd.DataFrame(data, columns=["delivery_time"])
    model = IsolationForest(contamination=0.05)
    df["anomaly"] = model.fit_predict(df[["delivery_time"]])
    anomalies = df[df["anomaly"] == -1]
    return JSONResponse(content=anomalies.to_dict(orient="records"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
