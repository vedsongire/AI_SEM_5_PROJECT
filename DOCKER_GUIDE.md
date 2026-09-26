# 🐳 K.I.S.A.N. AI - Docker Setup & Sharing Guide

This guide explains how to containerize and easily share the entire **K.I.S.A.N. AI** platform with your friend or collaborator so they can run it on any machine (Windows, macOS, or Linux) with **zero environment setup** (no Python or virtualenv installation needed).

---

## 🚀 Method 1: Share Project Folder & Run via Docker Compose (Recommended)

This is the cleanest and fastest way to collaborate.

### Step 1: Prepare the Project to Share
Compress the project folder into a `.zip` file (or share via Google Drive / Pen Drive / GitHub).
> **Tip:** You do **not** need to include `.venv/` (your local virtual environment) because Docker creates its own isolated Linux environment with all dependencies pre-installed.

### Step 2: What Your Friend Needs to Do
Your friend only needs **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** installed and running.

1. **Unzip** the project folder and open a terminal (PowerShell, Command Prompt, or Bash) in the project directory:
   ```bash
   cd AI_SEM_5_PROJECT
   ```

2. **Start the application with one command:**
   ```bash
   docker compose up --build
   ```

3. **Open the Web Dashboard:**
   Visit **[http://localhost:5000](http://localhost:5000)** in any web browser.

4. **To run in the background (detached mode):**
   ```bash
   docker compose up -d
   ```

5. **To stop the application:**
   ```bash
   docker compose down
   ```

---

## 📦 Method 2: Export a Self-Contained Docker Image (`.tar` Archive)

If your friend has a slow internet connection and you want to give them a pre-built image so they don't even need to download Python packages:

### On Your Computer (Exporting):
1. Build the Docker image locally:
   ```bash
   docker build -t kisan-ai:latest .
   ```

2. Save the image to a tarball archive:
   ```bash
   docker save -o kisan-ai-image.tar kisan-ai:latest
   ```

3. Give the `kisan-ai-image.tar` file (and the `data/` folder) to your friend via a USB flash drive or cloud storage.

### On Your Friend's Computer (Importing & Running):
1. Load the Docker image into their local Docker:
   ```bash
   docker load -i kisan-ai-image.tar
   ```

2. Run the container:
   ```bash
   docker run -d -p 5000:5000 -v "${PWD}/data:/app/data" --name kisan-ai-app kisan-ai:latest
   ```

3. Open **[http://localhost:5000](http://localhost:5000)**.

---

## ☁️ Method 3: Share via Docker Hub (Cloud Registry)

If you have a free [Docker Hub](https://hub.docker.com/) account:

1. **Tag and push the image:**
   ```bash
   docker login
   docker tag kisan-ai:latest <your-dockerhub-username>/kisan-ai:latest
   docker push <your-dockerhub-username>/kisan-ai:latest
   ```

2. **Your friend simply runs:**
   ```bash
   docker run -d -p 5000:5000 <your-dockerhub-username>/kisan-ai:latest
   ```

---

## ⚙️ Configuration & Features

### Live Data & Volume Mounts
The `docker-compose.yml` mounts local directories into the container:
- `./data:/app/data` — Gives the container instant access to all APMC records and the 300+ crop CSV files without bloating the image.
- `./models:/app/models` — Machine learning weights (p10, p50, p90 quantile regressors).
- `./config:/app/config` — Configuration and optional `.env` API keys.

### Viewing Logs & Status
- View real-time logs:
  ```bash
  docker compose logs -f
  ```
- Check container status:
  ```bash
  docker ps
  ```

### Container Health Check
The container includes an automated health check that queries `http://localhost:5000/` every 30 seconds to ensure the multi-agent Flask application is responsive.
