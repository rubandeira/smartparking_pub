import cv2
import json
import numpy as np
import glob
import os
import time
import math
import psycopg2 
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from datetime import datetime
from dotenv import load_dotenv

# --------------------------------------------
# 1. CONFIGURAÇÃO E CONEXÃO
# --------------------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


MODEL_NAME = "yolo26s.pt"      

def executar_sql(query, params=None, fetch=False):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = None
        if fetch: result = cursor.fetchall()
        conn.commit()
        cursor.close()
        return result
    except Exception as e:
        print(f" Erro SQL: {e}")
        return None
    finally:
        if conn: conn.close()

# --------------------------------------------
# 2. CAMINHOS
# --------------------------------------------
BASE_DIR = r"/home/rasp-1/Desktop/parking_2.0" 
FRAMES_DIR = os.path.join(BASE_DIR, "media", "frames")

JSON_ROI = os.path.join(BASE_DIR, "media", "estacionamento_jsons", "zona_interesse.json")
JSON_VAGAS = os.path.join(BASE_DIR, "media", "estacionamento_jsons", "vagas_linha.json")

YOLO_MODEL = os.path.join(BASE_DIR, MODEL_NAME)
VEHICLE_CLASSES = [2, 3, 5, 7] 

# --------------------------------------------
# 3. FUNÇÕES DE BASE DE DADOS
# --------------------------------------------
def inicializar_lugares_db(prefixo, quantidade, tipo):
    print(f" A verificar {quantidade} lugares '{tipo}'...")
    for i in range(quantidade):
        spot_id = f"{prefixo}-{i+1:02d}"
        executar_sql("INSERT INTO lugares (id, tipo, estado) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING;", (spot_id, tipo, "livre"))

def carregar_estado_inicial(prefixo_filtro):
    query = "SELECT id, estado, carro_atual_id FROM lugares WHERE id LIKE %s;"
    resultados = executar_sql(query, (f"{prefixo_filtro}%",), fetch=True)
    estado_sync = {}
    if resultados:
        for row in resultados:
            spot_id, estado_texto, car_id_db = row
            is_occupied = (estado_texto == 'ocupado')
            estado_sync[spot_id] = {'occupied': is_occupied, 'car_id': car_id_db}
    return estado_sync

def registar_entrada(spot_id, car_id):
    car_id_int = int(car_id)
    print(f" OCUPAÇÃO: {spot_id} (Carro {car_id_int})")
    agora = datetime.utcnow()
    executar_sql("UPDATE lugares SET estado = 'ocupado', carro_atual_id = %s, ultimo_update = %s WHERE id = %s;", (car_id_int, agora, spot_id))
    executar_sql("INSERT INTO historico (lugar_id, carro_id, entrada) VALUES (%s, %s, %s);", (spot_id, car_id_int, agora))

def registar_saida(spot_id, car_id):
    print(f" LIVRE: {spot_id}")
    agora = datetime.utcnow()
    executar_sql("UPDATE lugares SET estado = 'livre', carro_atual_id = NULL, ultimo_update = %s WHERE id = %s;", (agora, spot_id))
    if car_id:
        res = executar_sql("SELECT id, entrada FROM historico WHERE lugar_id = %s AND carro_id = %s AND saida IS NULL ORDER BY entrada DESC LIMIT 1;", (spot_id, int(car_id)), fetch=True)
        if res:
            hist_id, entrada_dt = res[0]
            if entrada_dt.tzinfo is not None: entrada_dt = entrada_dt.replace(tzinfo=None)
            duracao = int((agora - entrada_dt).total_seconds())
            executar_sql("UPDATE historico SET saida = %s, duracao_segundos = %s WHERE id = %s;", (agora, duracao, hist_id))

# --------------------------------------------
# 4. INICIALIZAÇÃO E GEOMETRIA
# --------------------------------------------
def get_feet_center(box):
    """ Devolve apenas o ponto central da base (pés do veículo) """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)

def load_zones(json_path):
    if not os.path.exists(json_path): return [], []
    with open(json_path, "r") as f: data = json.load(f)
    return [Polygon(d) for d in data], [np.array(d, np.int32) for d in data]

print(f"⏳ A carregar modelo {MODEL_NAME}...")
model = YOLO(YOLO_MODEL) 

roi_polys, _ = load_zones(JSON_ROI)
from shapely.ops import unary_union
roi_master = unary_union(roi_polys) if roi_polys else None

vagas_polys, vagas_points = load_zones(JSON_VAGAS)


estado_lugares = {}
buffers_vagas = {} 

historico_posicoes = {}
LIMITE_MOVIMENTO = 5 

for i in range(len(vagas_polys)): 
    spot_id = f"A-{i+1:02d}"
    estado_lugares[spot_id] = {'occupied': False, 'car_id': None}
    buffers_vagas[spot_id] = {'contagem_entrada': 0, 'contagem_saida': 0}

if DATABASE_URL:
    try:
        inicializar_lugares_db("A", len(vagas_polys), "normal")
        print(" A recuperar estados das vagas normais da Base de Dados...")
        dados_A = carregar_estado_inicial("A")
        if dados_A:
            for spot_id, data in dados_A.items():
                if spot_id in estado_lugares:
                    estado_lugares[spot_id] = data
        print("✔ Sistema Pronto.")
    except Exception as e: 
        print(f"❌ Erro no arranque da DB: {e}")

last_processed = None
print("\n Monitorização Ativa (Footpoint + Filtro Cinemático).")

def jpeg_completo(path):
    try:
        with open(path, "rb") as f:
            f.seek(-2, os.SEEK_END)
            return f.read() == b"\xff\xd9"
    except:
        return False

# --------------------------------------------
# 5. LOOP PRINCIPAL
# --------------------------------------------
while True:
    try:
        files = glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))
        if not files: continue
        newest = max(files, key=os.path.getmtime)
        if newest == last_processed:
            time.sleep(0.1)
            continue
        if not jpeg_completo(newest): continue
        frame = cv2.imread(newest)
        if frame is None or frame.size == 0: continue
        last_processed = newest
    except (FileNotFoundError, OSError):
        continue

  
    results = model.track(
        frame, 
        tracker="bytetrack.yaml", 
        persist=True, 
        verbose=False, 
        classes=VEHICLE_CLASSES 
    )[0]
    
    boxes_data = [] 
    if results.boxes and results.boxes.id is not None:
        raw_boxes = results.boxes.xyxy.cpu().numpy()
        raw_ids = results.boxes.id.cpu().numpy().astype(int)
        
        for b, t in zip(raw_boxes, raw_ids):
            ponto_central = get_feet_center(b)
            
            
            if roi_master and not roi_master.contains(Point(ponto_central)): continue
            
            
            is_stopped = False
            if t in historico_posicoes:
                ponto_anterior = historico_posicoes[t]
                distancia = math.dist(ponto_anterior, ponto_central)
                if distancia < LIMITE_MOVIMENTO:
                    is_stopped = True
            
            historico_posicoes[t] = ponto_central 
            
            boxes_data.append({
                'box': b, 
                'id': t, 
                'center': ponto_central,
                'is_stopped': is_stopped
            })

    # ======================================================
    # LÓGICA DE VAGAS NORMAIS (HISTERESE + CINEMÁTICA)
    # ======================================================
    FRAMES_PARA_OCUPAR = 5
    FRAMES_PARA_LIBERTAR = 10

    detecoes_neste_frame = {}
    info_carros = {}
    
    
    for i, poly in enumerate(vagas_polys):
        spot_id = f"A-{i+1:02d}"
        for data in boxes_data:
            info_carros[data['id']] = data['is_stopped']
            if poly.contains(Point(data['center'])):
                detecoes_neste_frame[spot_id] = data['id']
                break
    
    
    for spot_id in [k for k in estado_lugares if k.startswith("A")]:
        state = estado_lugares[spot_id]
        carro_detectado = detecoes_neste_frame.get(spot_id)
        buffer = buffers_vagas[spot_id]

        if carro_detectado:
            
            if info_carros.get(carro_detectado, False):
                buffer['contagem_entrada'] += 1
                buffer['contagem_saida'] = 0 
                
                if not state['occupied'] and buffer['contagem_entrada'] >= FRAMES_PARA_OCUPAR:
                    registar_entrada(spot_id, carro_detectado)
                    estado_lugares[spot_id] = {'occupied': True, 'car_id': carro_detectado}
            else:
                
                buffer['contagem_entrada'] = 0
                buffer['contagem_saida'] = 0
        else:
            buffer['contagem_entrada'] = 0 
            if state['occupied']:
                buffer['contagem_saida'] += 1
                if buffer['contagem_saida'] >= FRAMES_PARA_LIBERTAR:
                    registar_saida(spot_id, state['car_id'])
                    estado_lugares[spot_id] = {'occupied': False, 'car_id': None}
                    buffer['contagem_saida'] = 0
