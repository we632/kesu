import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ===== 可配置项 =====
# 你上线后建议设成你的域名，比如 https://img.yourdomain.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
ALLOWED_MIME_PREFIX = "image/"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Image Uploader")

# ✅ 允许跨域：你前端在另一个域名/端口也能调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 上线后你可以改成只允许你的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 静态访问：/uploads/xxxx.png
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith(ALLOWED_MIME_PREFIX):
        raise HTTPException(status_code=400, detail="Not an image")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        # 没扩展名或奇怪扩展名：根据 content-type 给个合理默认
        if file.content_type == "image/jpeg":
            ext = ".jpg"
        elif file.content_type == "image/webp":
            ext = ".webp"
        elif file.content_type == "image/gif":
            ext = ".gif"
        else:
            ext = ".png"

    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)

    content = await file.read()
    # 可加大小限制：比如 > 6MB 拒绝
    if len(content) > 6 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (>6MB)")

    with open(path, "wb") as f:
        f.write(content)

    url_path = f"/uploads/{name}"
    # 返回绝对 URL（更稳），否则返回相对路径
    if PUBLIC_BASE_URL:
        return {"url": f"{PUBLIC_BASE_URL}{url_path}"}
    return {"url": url_path}
from fastapi.responses import FileResponse

@app.get("/")
def home():
    return FileResponse("index.html")
