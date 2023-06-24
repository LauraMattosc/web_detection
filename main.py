from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Rota inicial para renderizar a página HTML
@app.route('/')
def index():
    return render_template('index.html')

# Rota de streaming de vídeo
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Função para gerar os frames do vídeo
def generate_frames():
    # Inicializar a câmera
    cap = cv2.VideoCapture(0)  # Use 0 para câmera padrão do sistema

    while True:
        # Ler o próximo frame da câmera
        ret, frame = cap.read()

        if ret:
            # Converter o frame para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar os rostos no frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Desenhar retângulos ao redor dos rostos detectados
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Converter o frame em formato de bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Gerar o frame como um bloco multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# Executar o aplicativo Flask
if __name__ == '__main__':
    app.run(debug=True)
