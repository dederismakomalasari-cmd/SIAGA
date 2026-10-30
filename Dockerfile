# Gunakan image dasar Python resmi
FROM python:3.11-slim

# Atur direktori kerja
WORKDIR /app

# Hugging Face Spaces mengharuskan container dijalankan sebagai non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Instal dependensi sistem yang diperlukan OpenCV
# Meskipun menggunakan opencv-python-headless, terkadang tetap memerlukan beberapa library dasar
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements.txt dan instal dependensi Python
# Menggunakan --chown=user untuk memastikan izin file
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin sisa kode proyek
COPY --chown=user . .

# Ekspos port yang digunakan Hugging Face Spaces (default 7860)
EXPOSE 7860

# Perintah startup: port harus 7860, dan jalur aplikasi disesuaikan dengan proyek Anda
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT