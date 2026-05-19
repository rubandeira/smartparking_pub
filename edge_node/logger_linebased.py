# -*- coding: utf-8 -*-
# ============================================
# SCRIPT 1: UNIVERSAL DATA LOGGER (ANY RESOLUTION)
# ============================================
import cv2
import pandas as pd
import glob
import os
import time
from ultralytics import YOLO


FRAMES_DIR = "/home/rasp-1/Desktop/parking_2.0/media/frames_china"
CSV_OUT = "historico_carros_linebased_china.csv"
MODEL_NAME = "yolo26n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7] 

if not os.path.exists(CSV_OUT):
    df_vazio = pd.DataFrame(columns=['timestamp', 'track_id', 'cx', 'cy', 'w', 'h'])
    df_vazio.to_csv(CSV_OUT, index=False)

print(f"A iniciar Logger Universal com {MODEL_NAME}...")
model = YOLO(MODEL_NAME)

last_processed = None
ultimo_registo_time = time.time()


SESSION_OFFSET = int(time.time())

def jpeg_completo(path):
    try:
        with open(path, "rb") as f:
            f.seek(-2, os.SEEK_END)
            return f.read() == b"\xff\xd9"
    except:
        return False

print("A escutar a pasta de frames (Grava 1 vez por segundo)...")

while True:
    try:
        files = glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))
        if not files: 
            time.sleep(0.1)
            continue
        
        files = [f for f in files if os.path.exists(f)]
        if not files:
            continue

        newest = max(files, key=os.path.getmtime)
        if newest == last_processed or not jpeg_completo(newest):
            time.sleep(0.1)
            continue
            
        frame = cv2.imread(newest)
        if frame is None: 
            continue
            
        last_processed = newest
        
        agora = time.time()
        if agora - ultimo_registo_time >= 1.0:
            
            results = model.track(
                frame, 
                tracker="botsort.yaml", 
                persist=True, 
                verbose=False, 
                classes=VEHICLE_CLASSES,
                imgsz=1280, 
                conf=0.15
            )[0]
            
            novos_dados = []
            if results.boxes and results.boxes.id is not None:
                boxes = results.boxes.xywh.cpu().numpy() 
                
             
                ids = results.boxes.id.cpu().numpy().astype(int) + SESSION_OFFSET
                
                for b, t in zip(boxes, ids):
                    cx, cy, w, h = b
                    novos_dados.append({
                        'timestamp': agora, 
                        'track_id': t, 
                        'cx': cx, 
                        'cy': cy, 
                        'w': w, 
                        'h': h
                    })
            
            if novos_dados:
                df_temp = pd.DataFrame(novos_dados)
                df_temp.to_csv(CSV_OUT, mode='a', header=False, index=False)
                print(f"[{time.strftime('%H:%M:%S')}] Registados {len(novos_dados)} veiculos (Resolucao detetada: {frame.shape[1]}x{frame.shape[0]})")
            
            ultimo_registo_time = agora

    except FileNotFoundError:
        pass 
    except Exception as e:
        print(f"Erro no loop: {e}")
        time.sleep(0.1)
