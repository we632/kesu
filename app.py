

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


# Gemini API key：推荐在环境变量里设置 GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

app = FastAPI(title="Complaint Template OCR API (Gemini)")

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


async def _extract_with_gemini(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """用 Gemini Vision 提取 orders + reason，返回 dict。"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    # 官方 Python SDK：google-genai
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    system_hint = (
        "You are a strict information extractor. "
        "Return ONLY valid JSON. No markdown, no extra text."
    )
    user_prompt = (
        "请从这张截图中提取并只输出 JSON（不要解释、不要 markdown）。\n\n"
        "输出格式：\n"
        '{ "orders": ["SWX..."], "text": "内部备注的完整文字（包含门禁密码/48小时等），去掉所有SWX订单号，按原顺序输出" }\n\n'
        "规则：\n"
        "1) orders：提取所有以 SWX 开头的订单号（可能有空格），输出时去掉空格。\n"
        "2) text：只保留“内部备注”后面的内容；把所有 SWX 订单号从 text 里移除；其余文字尽量原样保留。\n"
        "3) 只输出 JSON。\n"
    )


    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    # 小贴士：单图+文本时，把“图”放前面，“提示词”放后面更稳（官方也这么建议）
    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[image_part, f"{system_hint}\n\n{user_prompt}"],
    )

    text_out = (resp.text or "").strip()

    # 解析 JSON（只取第一个 {...}）
    import json
    m = re.search(r"\{.*\}", text_out, re.S)
    if not m:
        raise RuntimeError(f"Model did not return JSON: {text_out[:200]}")

    data = json.loads(m.group(0))

    # ✅ 1) text：从 JSON 里取（模型负责“内部备注全文”）
    text = data.get("text") or data.get("reason") or ""
    text = _normalize_cn_spaces(str(text))

    # ✅ 2) orders：不要依赖模型，直接从 text_out 抓 SWX（最稳）
    orders = re.findall(r"SWX\s*\d+", text_out, flags=re.I)
    orders = [re.sub(r"\s+", "", o).upper() for o in orders]
    orders = [re.sub(r"[^A-Za-z0-9]", "", o) for o in orders]  # 去掉可能的标点
    orders = [o for o in orders if re.fullmatch(r"SWX\d+", o, flags=re.I)]

    return {"orders": orders, "text": text, "reason": text}




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