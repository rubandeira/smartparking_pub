import os
import time
import cv2
import yt_dlp
from datetime import datetime

YOUTUBE_URL = "https://www.youtube.com/watch?v=EPKWu223XEg"

FRAMES_DIR = "/home/rasp-1/Desktop/parking_2.0/media/frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

INTERVALO = 0.5
MAX_FRAMES = 30

# Otimização: Pedir apenas vídeo (poupa internet e CPU) e limitar a 720p se possível
ydl_opts = {
    "quiet": True,
    "no_warnings": True,
    "format": "bestvideo[height<=720]/bestvideo+bestaudio/best", 
}

def get_stream_url():
    print("🔄 A renovar URL do stream...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(YOUTUBE_URL, download=False)
            return info["url"]
    except Exception as e:
        print(f"❌ Erro ao obter URL: {e}")
        return None

def clean_old_frames():
    try:
        files = sorted(
            [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".jpg", ".png"))]
        )
        # Se tiver mais ficheiros que o MAX, apaga os mais antigos
        if len(files) > MAX_FRAMES:
            for f in files[:-MAX_FRAMES]:
                try:
                    os.remove(os.path.join(FRAMES_DIR, f))
                except OSError:
                    pass
    except Exception as e:
        print(f"Erro ao limpar frames: {e}")

# =============================================================
# INICIALIZAÇÃO
# =============================================================

stream_url = get_stream_url()
while not stream_url: # Garante que temos URL antes de começar
    time.sleep(5)
    stream_url = get_stream_url()

cap = cv2.VideoCapture(stream_url)

# Tenta diminuir o tamanho do buffer interno do OpenCV (nem sempre funciona em todos os backends, mas ajuda)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_save_time = time.time()
fail_count = 0

# =============================================================
# LOOP PRINCIPAL
# =============================================================

print("🚀 Iniciando captura em tempo real...")

while True:
    # 1. Lê o frame IMEDIATAMENTE (sem sleeps antes)
    ret, frame = cap.read()

    # 2. Se o frame falhar, reconecta
    if not ret or frame is None:
        print("⚠ Stream caiu ou frame inválido. A reconectar...")
        cap.release()
        time.sleep(2) # Aqui o sleep é ok porque estamos parados
        
        stream_url = get_stream_url()
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            time.sleep(5)
        continue

    # 3. Lógica de Tempo: Só guarda se já passou o INTERVALO
    current_time = time.time()
    if current_time - last_save_time >= INTERVALO:
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(FRAMES_DIR, f"yt_frame_{timestamp}.jpg")
        
        # Guarda o frame
        cv2.imwrite(path, frame)
        print(f"✔ Frame guardado: {path}")

        # Limpa antigos
        clean_old_frames()

        # Atualiza o relógio
        last_save_time = current_time
    
    # IMPORTANTE: Não há time.sleep() aqui no final! 
    # O loop corre à velocidade máxima do vídeo (ex: 30fps) para esvaziar o buffer,
    # mas o 'if' acima garante que só guardamos imagens a cada 0.5s.
