import pandas as pd
import numpy as np
import cv2
import json
import os
import math
import glob
import random

# --- CONFIGURAÇÃO ---
INPUT_CSV = "historico_carros_linebased_colingwood.csv"
OUTPUT_JSON = "/home/rasp-1/Desktop/parking_2.0/media/estacionamento_jsons/vagas_teste.json"
FRAMES_DIR = "/home/rasp-1/Desktop/parking_2.0/media/frames_china"
LARGURA, ALTURA = 1920, 1080

# --- PARÂMETROS ---
MIN_FRAMES_PARADO = 200
THRESHOLD_VAL = 35
COMPRIMENTO_MIN_FACTOR = 1.8  
COMPRIMENTO_MAX_FACTOR = 2.2  

# =======================================================
# 1. GERAÇÃO DO MAPA DE CALOR
# =======================================================
print(" A carregar dados e a gerar mapa de calor...")
if not os.path.exists(INPUT_CSV):
    print(f" Erro: {INPUT_CSV} não encontrado.")
    exit()

df = pd.read_csv(INPUT_CSV)
ids_validos = df['track_id'].value_counts()[df['track_id'].value_counts() > MIN_FRAMES_PARADO].index
df_agrupado = df[df['track_id'].isin(ids_validos)].groupby('track_id').median().reset_index()

heatmap = np.zeros((ALTURA, LARGURA), dtype=np.float32)
for _, row in df_agrupado.iterrows():
    cx, cy, w, h = int(row['cx']), int(row['cy']), int(row['w']), int(row['h'])
    y_chao = int(cy + (h / 2))
    l_lin = int(w * 0.8)
    x1, x2 = max(0, int(cx - l_lin//2)), min(LARGURA, int(cx + l_lin//2))
    temp_mask = np.zeros((ALTURA, LARGURA), dtype=np.float32)
    cv2.line(temp_mask, (x1, y_chao), (x2, y_chao), color=1, thickness=15)
    heatmap += temp_mask

heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, binary_map = cv2.threshold(heatmap_norm, THRESHOLD_VAL, 255, cv2.THRESH_BINARY)
binary_map = cv2.dilate(binary_map, np.ones((11,11), np.uint8), iterations=1)
binary_map = cv2.erode(binary_map, np.ones((5,5), np.uint8), iterations=1)

# =======================================================
# 1.5 OUTPUT DO HEATMAP (AGORA GUARDA AS IMAGENS)
# =======================================================
print("\n  A guardar e a mostrar o Heatmap gerado...")


cv2.imwrite("heatmap_binario_china.jpg", binary_map)


heatmap_cores = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_INFERNO)
cv2.imwrite("heatmap_cores_china.jpg", heatmap_cores)
print(" Heatmaps guardados com sucesso na pasta do script!")

print("Pressiona qualquer tecla na janela da imagem para iniciar a calibração manual.")
cv2.imshow("Heatmap Preview", binary_map)
cv2.waitKey(0) 
cv2.destroyWindow("Heatmap Preview") 

# =======================================================
# 2. SELEÇÃO DE FUNDO E EXTRAÇÃO DE VAGAS
# =======================================================
frames_list = glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))
img_validation = cv2.resize(cv2.imread(random.choice(frames_list)), (LARGURA, ALTURA)) if frames_list else np.zeros((ALTURA, LARGURA, 3), np.uint8)

contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_candidates = []
for cnt in contours:
    if 150 < cv2.contourArea(cnt) < 60000:
        rect = cv2.minAreaRect(cnt)
        valid_candidates.append({'center': rect[0], 'rect_base': rect})

# =======================================================
# 3. INTERFACE DE AJUSTE DIRECIONAL (WASD + XZ + QE)
# =======================================================
lugares_finais = []
print("\n" + "="*70)
print(" FASE 1: CALIBRAÇÃO AUTOMÁTICA")
print("W / S : CIMA (Aumentar/Diminuir)   |   X / Z : BAIXO (Aumentar/Diminuir)")
print("D / A : DIREITA (Aumentar/Diminuir)|   Q / E : ESQUERDA (Aumentar/Diminuir)")
print("Y : CONFIRMAR                      |   N : REJEITAR")
print("="*70 + "\n")

for i, cand in enumerate(valid_candidates):
    cx, cy = cand['center']
    (w_base, h_base), angle = cand['rect_base'][1], cand['rect_base'][2]
    
    if w_base >= h_base:
        curr_w, curr_h = w_base * 1.1, h_base * 2.0
    else:
        curr_w, curr_h = w_base * 2.0, h_base * 1.1

    while True:
        rect_adj = ((cx, cy), (curr_w, curr_h), angle)
        box = np.int32(cv2.boxPoints(rect_adj))
        
        temp_img = img_validation.copy()
        cv2.drawContours(temp_img, [box], 0, (0, 255, 255), 3) 
        
        cv2.rectangle(temp_img, (20, 20), (940, 210), (0,0,0), -1)
        cv2.putText(temp_img, f"AJUSTE INDEPENDENTE - VAGA {i+1}/{len(valid_candidates)}", (40, 60), 1, 2, (0, 255, 255), 2)
        cv2.putText(temp_img, "W(+)/S(-): Topo | X(+)/Z(-): Base", (40, 100), 1, 1.2, (255, 255, 255), 1)
        cv2.putText(temp_img, "D(+)/A(-): Dir  | Q(+)/E(-): Esq", (40, 140), 1, 1.2, (255, 255, 255), 1)
        cv2.putText(temp_img, "[Y] GRAVAR  [N] DESCARTAR", (40, 185), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("Calibrador de Vagas 4.0", temp_img)
        key = cv2.waitKey(0) & 0xFF
        
        step = 6
        rad = math.radians(angle)
        v_dir = np.array([-math.sin(rad), math.cos(rad)]) 
        h_dir = np.array([math.cos(rad), math.sin(rad)])

        if key == ord('w'): curr_h += step; cx -= v_dir[0]*(step/2); cy -= v_dir[1]*(step/2)
        elif key == ord('s'): 
            if curr_h > step: curr_h -= step; cx += v_dir[0]*(step/2); cy += v_dir[1]*(step/2)
        elif key == ord('x'): curr_h += step; cx += v_dir[0]*(step/2); cy += v_dir[1]*(step/2)
        elif key == ord('z'): 
            if curr_h > step: curr_h -= step; cx -= v_dir[0]*(step/2); cy -= v_dir[1]*(step/2)
        elif key == ord('d'): curr_w += step; cx += h_dir[0]*(step/2); cy += h_dir[1]*(step/2)
        elif key == ord('a'): 
            if curr_w > step: curr_w -= step; cx -= h_dir[0]*(step/2); cy -= h_dir[1]*(step/2)
        elif key == ord('q'): curr_w += step; cx -= h_dir[0]*(step/2); cy -= h_dir[1]*(step/2)
        elif key == ord('e'): 
            if curr_w > step: curr_w -= step; cx += h_dir[0]*(step/2); cy += h_dir[1]*(step/2)
            
        elif key == ord('y'):
            lugares_finais.append(box.tolist())
            cv2.drawContours(img_validation, [box], 0, (0, 255, 0), 2)
            cv2.putText(img_validation, f"L{len(lugares_finais)}", (int(cx), int(cy)), 1, 1, (0,255,0), 2)
            break
        elif key == ord('n'):
            break
        elif key == 27:
            cv2.destroyAllWindows()
            exit()

cv2.destroyWindow("Calibrador de Vagas 4.0")

# =======================================================
# 4. MODO DE DESENHO MANUAL (PREENCHER FALHAS)
# =======================================================
print("\n" + "="*70)
print("  FASE 2: MODO DE DESENHO MANUAL")
print("Clica em 4 cantos para desenhar as vagas que faltam no parque.")
print(" - [Y] Confirmar a vaga que acabaste de desenhar")
print(" - [C] Limpar os pontos para recomeçar")
print(" - [ESC] Terminar e Gravar o ficheiro JSON final")
print("="*70 + "\n")

pontos_manuais = []

def draw_polygon(event, x, y, flags, param):
    global pontos_manuais
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pontos_manuais) < 4:
            pontos_manuais.append([x, y])

cv2.namedWindow("Desenho Manual")
cv2.setMouseCallback("Desenho Manual", draw_polygon)

while True:
    temp_img = img_validation.copy()
    
    
    for f_box in lugares_finais:
        cv2.drawContours(temp_img, [np.array(f_box)], 0, (0, 255, 0), 2)
        
    
    for pt in pontos_manuais:
        cv2.circle(temp_img, tuple(pt), 5, (0, 0, 255), -1)
        
    
    if len(pontos_manuais) > 1:
        cv2.polylines(temp_img, [np.array(pontos_manuais)], False, (0, 255, 255), 2)
        
    
    if len(pontos_manuais) == 4:
        cv2.polylines(temp_img, [np.array(pontos_manuais)], True, (0, 255, 255), 2)
        cv2.rectangle(temp_img, (20, 20), (700, 100), (0,0,0), -1)
        cv2.putText(temp_img, "[Y] GRAVAR VAGA  |  [C] LIMPAR", (40, 65), 1, 1.5, (0, 255, 255), 2)
        
    cv2.imshow("Desenho Manual", temp_img)
    key = cv2.waitKey(20) & 0xFF
    
    if len(pontos_manuais) == 4:
        if key == ord('y'):
            lugares_finais.append(pontos_manuais.copy())
            pontos_manuais = []
        elif key == ord('c'):
            pontos_manuais = []
    else:
        if key == ord('c'):
            pontos_manuais = []
            
    if key == 27: # ESC
        break

cv2.destroyAllWindows()

# Gravar JSON
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(lugares_finais, f)

print(f"🎉 Processo concluído! {len(lugares_finais)} lugares guardados em JSON.")
