

import os
import re
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ---------- 配置 ----------
# 你的结构是：repo_root/backend/app.py 以及 repo_root/frontend/index.html
BASE_DIR = Path(__file__).resolve().parent  # backend/
CANDIDATE_1 = BASE_DIR.parent / "frontend"  # repo_root/frontend
CANDIDATE_2 = BASE_DIR / "frontend"         # backend/frontend

FRONTEND_DIR = CANDIDATE_1 if CANDIDATE_1.exists() else CANDIDATE_2
print("FRONTEND_DIR =", FRONTEND_DIR)
print("EXISTS =", FRONTEND_DIR.exists())


# Groq API key：推荐在环境变量里设置 GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# 可通过环境变量覆盖模型，避免模型下线导致服务不可用
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "groq").strip().lower()

app = FastAPI(title="Complaint Template OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你自己部署后可以改成你的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def home():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse(
        {"ok": False, "error": f"frontend not found. FRONTEND_DIR={FRONTEND_DIR}"},
        status_code=404,
    )



def _normalize_cn_spaces(s: str) -> str:
    if not s:
        return ""
    t = s
    t = re.sub(r"[\u3000\u00A0\u2000-\u200B\u202F\u205F\uFEFF]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # 去掉：汉字/数字/字母 之间的空格
    t = re.sub(r"([\u4e00-\u9fffA-Za-z0-9])\s+([\u4e00-\u9fffA-Za-z0-9])", r"\1\2", t)
    # 汉字与中文标点空格
    t = re.sub(r"([\u4e00-\u9fff])\s+([，。；：、])", r"\1\2", t)
    t = re.sub(r"([，。；：、])\s+([\u4e00-\u9fff])", r"\1\2", t)
    return t.strip()


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _compress_to_jpeg(image_bytes: bytes, max_side: int = 1280, quality: int = 72) -> bytes:
    """可选压缩：如果没装 pillow，会原样返回。"""
    try:
        from PIL import Image  # type: ignore
        import io

        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return image_bytes


async def _extract_with_provider(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """用 Groq Vision 提取 orders + reason，返回 dict。"""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    import base64
    import json
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    prompt = (
        "请从这张截图中提取并只输出 JSON（不要解释、不要 markdown）。\n\n"
        "输出格式：\n"
        '{ "orders": ["SWX..."], "text": "内部备注的完整文字（包含门禁密码/48小时等），去掉所有SWX订单号，按原顺序输出" }\n\n'
        "规则：\n"
        "1) orders：提取所有以 SWX 开头的订单号（可能有空格），输出时去掉空格。\n"
        "2) text：只保留‘内部备注’后面的内容；把所有 SWX 订单号从 text 里移除；其余文字尽量原样保留。\n"
        "3) 只输出 JSON。\n"
    )

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    fallback_models = [
        GROQ_VISION_MODEL,
        "meta-llama/llama-4-maverick-17b-128e-instruct",
    ]

    last_error = None
    text_out = ""
    for model_name in _dedupe_keep_order(fallback_models):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            text_out = (resp.choices[0].message.content or "").strip()
            if text_out:
                break
        except Exception as e:
            last_error = e

    if not text_out:
        if last_error:
            raise RuntimeError(f"Groq request failed: {last_error}") from last_error
        raise RuntimeError("Groq request failed: empty response")

    m = re.search(r"\{.*\}", text_out, re.S)
    if not m:
        raise RuntimeError(f"Model did not return JSON: {text_out[:200]}")

    data = json.loads(m.group(0))
    text = _normalize_cn_spaces(str(data.get("text") or data.get("reason") or ""))

    orders = re.findall(r"SWX\s*\d+", text_out, flags=re.I)
    orders = [re.sub(r"\s+", "", o).upper() for o in orders]
    orders = [re.sub(r"[^A-Za-z0-9]", "", o) for o in orders]
    orders = [o for o in orders if re.fullmatch(r"SWX\d+", o, flags=re.I)]

    return {"orders": orders, "text": text, "reason": text}



async def _extract_with_gemini(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """兼容旧调用：项目已切到 Groq，这里保留同签名入口以降低分支冲突。"""
    return await _extract_with_provider(image_bytes, mime_type)


async def _extract_with_groq(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """兼容新调用：与 provider 实现保持一致。"""
    return await _extract_with_provider(image_bytes, mime_type)



def _extract_address_fallback(text: str) -> str:
    """
    从 OCR/模型返回的原始文本里兜底抓一个像美国地址的串：
    包含门牌号 + 街道 + (apt/unit可选) + 城市 + 州缩写 + 邮编
    """
    if not text:
        return ""

    t = " ".join(text.split())  # 压缩空白

    # 典型：900 Montgomery ave apt 506 Bryn Mawr PA 19010
    pat = re.compile(
        r"\b(\d{1,6}\s+[A-Za-z0-9.\- ]+?\s+"
        r"(?:(?:apt|apartment|unit|ste|suite|#)\s*\w+\s+)?"   # ✅ 这里包了一层 (?:(... )?) 才能 optional
        r"[A-Za-z.\- ]+?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?)\b",
        re.IGNORECASE
    )

    m = pat.search(t)
    return m.group(1).strip() if m else ""

@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_bytes = _compress_to_jpeg(image_bytes)

        mime_type = file.content_type or "image/jpeg"
        data = await _extract_with_gemini(image_bytes, mime_type)

        return {
            "ok": True,
            "orders": data.get("orders", []),
            "text": data.get("text", ""),
            "reason": data.get("text", "")
        }


    except Exception as e:
        return {"ok": False, "error": str(e)}