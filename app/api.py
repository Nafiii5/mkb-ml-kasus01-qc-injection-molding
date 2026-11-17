from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict_cycle

app = FastAPI(title="API Pengendalian Mutu Injection Molding", version="1.0.0")


class CycleInput(BaseModel):
    mesin: str = "IM-01"
    shift: str = "pagi"
    bahan: str = "PP"
    suhu_leleh_C: Optional[float] = 230.0
    suhu_cetakan_C: Optional[float] = 40.0
    tekanan_injeksi_bar: Optional[float] = 920.0
    kecepatan_injeksi_mm_s: Optional[float] = 80.0
    waktu_pendinginan_s: Optional[float] = 18.0
    waktu_siklus_s: Optional[float] = 32.0
    kelembaban_bahan_pct: Optional[float] = 0.02


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: CycleInput):
    try:
        return predict_cycle(data.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
