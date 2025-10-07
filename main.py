# main.py
import functions_framework
from cloudevents.http import CloudEvent

from google.cloud import firestore
from google.cloud import storage
import google.generativeai as genai

import requests
import json
import os
import logging
from PIL import Image, ExifTags
import io
import math
from datetime import datetime, timezone, timedelta

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Env / Clients ----------
os.environ["GOOGLE_GENAI_DISABLE_AUDIT_LOGS"] = "true"

db = None
storage_client = None
_GENAI_READY = False

def _ensure_clients():
    """Initialize Firestore, Storage, and Gemini client if not yet ready."""
    global db, storage_client, _GENAI_READY
    if db is None:
        db = firestore.Client()
    if storage_client is None:
        storage_client = storage.Client()
    if not _GENAI_READY:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            logger.info("[startup] Gemini configured")
        else:
            logger.warning("[startup] GEMINI_API_KEY missing – Gemini may fail")
        _GENAI_READY = True


# Try these in order until one works with generateContent
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
]

def _generate_with_available_model(parts):
    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(name)
            resp = model.generate_content(parts)
            logger.info(f"Gemini model used: {name}")
            return resp
        except Exception as e:
            msg = str(e)
            if any(s in msg.lower() for s in ["not found", "not supported", "404"]):
                logger.warning(f"Model {name} unavailable; trying next.")
                last_err = e
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("No Gemini model available.")


# ---------- Utilities ----------
def _report_id_from_subject(cloud_event: CloudEvent) -> str:
    subject = None
    try:
        subject = cloud_event.get("subject")
    except Exception:
        try:
            subject = cloud_event["subject"]
        except Exception:
            subject = None
    if isinstance(subject, str) and subject:
        return subject.split("/")[-1]
    data = cloud_event.data
    if isinstance(data, dict):
        resource = data.get("value", {}).get("name", "")
        if resource:
            return resource.split("/")[-1]
    return ""

def _ts_to_dt(ts):
    """Firestore TS or ISO or number -> aware datetime (UTC)."""
    if ts is None:
        return None
    if hasattr(ts, "to_datetime"):
        try:
            return ts.to_datetime().astimezone(timezone.utc)
        except Exception:
            pass
    if hasattr(ts, "toDate"):
        try:
            return ts.toDate().astimezone(timezone.utc)
        except Exception:
            pass
    if getattr(ts, "seconds", None) is not None:
        return datetime.fromtimestamp(ts.seconds, tz=timezone.utc)
    if isinstance(ts, (int, float)):
        if ts < 1e12:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def _extract_lat_lng(report_data):
    """Try several field shapes to get lat/lng."""
    loc = report_data.get("location") or {}
    lat = (loc.get("lat") or loc.get("latitude") or report_data.get("lat") or report_data.get("latitude"))
    lng = (loc.get("lng") or loc.get("lon") or loc.get("longitude") or report_data.get("lng") or report_data.get("lon") or report_data.get("longitude"))
    try:
        if lat is not None and lng is not None:
            return float(lat), float(lng)
    except Exception:
        pass
    return None, None

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _bbox(lat, lng, radius_m):
    # Approximate: 1 deg lat ~ 111,320 m ; 1 deg lon ~ 111,320 * cos(lat)
    dlat = radius_m / 111320.0
    dlng = radius_m / (111320.0 * max(0.1, math.cos(math.radians(lat))))
    return (lat - dlat, lat + dlat, lng - dlng, lng + dlng)


# ---------- Factor 1: Vision (existing) + Depth & AI-gen ----------
def _vision_flood_depth_aigen(image: Image.Image, user_depth_label: str | None):
    """
    Returns dict:
      {
        "isFloodWater": bool,
        "visionConfidence": float,
        "depthLabel": "minimal|moderate|severe|unknown",
        "depthAgreement": "match|near|mismatch|unknown",
        "aiGenLabel": "low|medium|high",
        "aiGenConfidence": float,
        "reasoning": "...",
      }
    """
    prompt = """You are a flood verification assistant. Analyze the photo and answer in JSON:

- isFloodWater: true/false (is this flooding in a developed area?)
- visionConfidence: 0..1 (how confident are you about that?)
- depthLabel: one of "minimal" (ankle-deep), "moderate" (knee-deep), "severe" (chest-deep), or "unknown"
- aiGenLabel: one of "low", "medium", "high" (likelihood the image is AI-generated)
- aiGenConfidence: 0..1
- reasoning: 2 short sentences explaining what visual cues you used

Return ONLY JSON like:
{
  "isFloodWater": true,
  "visionConfidence": 0.82,
  "depthLabel": "moderate",
  "aiGenLabel": "low",
  "aiGenConfidence": 0.15,
  "reasoning": "…"
}"""
    # Call Gemini
    resp = _generate_with_available_model([prompt, image])
    text = getattr(resp, "text", "") or ""
    if not text and getattr(resp, "candidates", None):
        parts = getattr(resp.candidates[0].content, "parts", [])
        text = parts[0].text if parts else ""

    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.endswith("```"):
        s = s[:-3]

    try:
        data = json.loads(s)
    except Exception:
        logger.warning("Gemini returned non-JSON; defaulting to low confidence")
        data = {}

    # normalize
    out = {
        "isFloodWater": bool(data.get("isFloodWater")),
        "visionConfidence": float(data.get("visionConfidence", 0.0)),
        "depthLabel": str(data.get("depthLabel") or "unknown").lower(),
        "aiGenLabel": str(data.get("aiGenLabel") or "low").lower(),
        "aiGenConfidence": float(data.get("aiGenConfidence", 0.0)),
        "reasoning": str(data.get("reasoning") or ""),
    }

    # compute depthAgreement vs user selection
    user = (user_depth_label or "").lower()
    model = out["depthLabel"]
    if user and model in ("minimal", "moderate", "severe"):
        if user == model:
            out["depthAgreement"] = "match"
        elif (user, model) in (("minimal","moderate"),("moderate","minimal"),
                               ("moderate","severe"),("severe","moderate")):
            out["depthAgreement"] = "near"
        else:
            out["depthAgreement"] = "mismatch"
    else:
        out["depthAgreement"] = "unknown"

    return out


# ---------- Factor 4: EXIF freshness ----------
def _parse_exif_freshness(pil_img: Image.Image, pin_lat: float | None, pin_lng: float | None, upload_dt_utc: datetime | None):
    """
    Returns dict with score in [0,1]:
      {
        "score": float,
        "hasExif": bool,
        "captureAt": "...iso..." or None,
        "gpsOk": True/False/None,
        "ageMinutes": int|None,
        "details": "...",
      }
    """
    result = {"score": 0.2, "hasExif": False, "captureAt": None, "gpsOk": None, "ageMinutes": None, "details": "no exif"}
    try:
        exif = pil_img.getexif()
        if not exif:
            return result
        # Map tag names
        label_map = {ExifTags.TAGS.get(k, str(k)): k for k in exif.keys()}
        dt_tag = label_map.get("DateTimeOriginal") or label_map.get("DateTime") or None
        gps_tag = label_map.get("GPSInfo") or None
        result["hasExif"] = True

        capture_dt = None
        if dt_tag:
            raw = exif.get(dt_tag)
            # "YYYY:MM:DD HH:MM:SS"
            try:
                capture_dt = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                try:
                    capture_dt = datetime.fromisoformat(str(raw).replace("Z","+00:00")).astimezone(timezone.utc)
                except Exception:
                    capture_dt = None
        if capture_dt:
            result["captureAt"] = capture_dt.isoformat()
            if upload_dt_utc:
                diff = abs((upload_dt_utc - capture_dt).total_seconds()) / 60.0
                result["ageMinutes"] = int(diff)

        gps_ok = None
        if gps_tag and pin_lat is not None and pin_lng is not None:
            gps = exif.get(gps_tag) or {}

            def _to_deg(vals):
                try:
                    d = float(vals[0][0]) / float(vals[0][1])
                    m = float(vals[1][0]) / float(vals[1][1])
                    s = float(vals[2][0]) / float(vals[2][1])
                    return d + m/60.0 + s/3600.0
                except Exception:
                    return None

            lat_vals = gps.get(2) or gps.get("GPSLatitude")
            lat_ref  = gps.get(1) or gps.get("GPSLatitudeRef")
            lon_vals = gps.get(4) or gps.get("GPSLongitude")
            lon_ref  = gps.get(3) or gps.get("GPSLongitudeRef")
            if lat_vals and lon_vals:
                exif_lat = _to_deg(lat_vals)
                exif_lng = _to_deg(lon_vals)
                if exif_lat is not None and exif_lng is not None:
                    if lat_ref in ("S","s"): exif_lat *= -1
                    if lon_ref in ("W","w"): exif_lng *= -1
                    dist = _haversine_m(pin_lat, pin_lng, exif_lat, exif_lng)
                    gps_ok = dist <= 300  # within 300 m
        result["gpsOk"] = gps_ok

        # Score
        score = 0.2  # baseline if EXIF exists but not helpful
        # time contribution
        if result["ageMinutes"] is not None:
            m = result["ageMinutes"]
            if m <= 30: score += 0.8
            elif m <= 120: score += 0.6
            elif m <= 1440: score += 0.3
            else: score += 0.05

        # gps contribution
        if gps_ok is True:
            score += 0.2
        elif gps_ok is False:
            score -= 0.2

        result["score"] = max(0.0, min(1.0, score))
        result["details"] = "parsed exif"
        return result
    except Exception as e:
        logger.warning(f"EXIF parse failed: {e}")
        return result


# ---------- Factor 6: Local cluster ----------
def _cluster_factor(pin_lat, pin_lng, window_minutes=90, radius_m=500):
    """
    Look at last window_minutes of verified/high-score reports near pin.
    Returns:
      {
        "score": 0..1,
        "uniqueUsers": int,
        "totalReports": int,
        "windowMinutes": int,
        "radiusM": int
      }
    """
    out = {"score": 0.1, "uniqueUsers": 0, "totalReports": 0, "windowMinutes": window_minutes, "radiusM": radius_m}
    if pin_lat is None or pin_lng is None:
        return out

    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

    # bounding box query (2 range filters -> composite index likely needed)
    minLat, maxLat, minLng, maxLng = _bbox(pin_lat, pin_lng, radius_m)

    base = db.collection("reports")
    q = (base.where("createdAt", ">=", since)
              .where("location.lat", ">=", minLat).where("location.lat", "<=", maxLat)
              .where("location.lng", ">=", minLng).where("location.lng", "<=", maxLng))

    try:
        docs = list(q.stream())
    except Exception as e:
        logger.warning(f"Cluster query failed (falling back to coarse): {e}")
        # Fallback: only time filter, distance check client-side
        docs = list(base.where("createdAt", ">=", since).stream())

    uniq = set()
    total = 0
    for d in docs:
        r = d.to_dict() or {}
        # distance filter if we used the coarse path
        lat, lng = _extract_lat_lng(r)
        if lat is None or lng is None:
            continue
        if _haversine_m(pin_lat, pin_lng, lat, lng) > radius_m:
            continue

        total += 1
        uid = (r.get("createdBy") or r.get("reporterId") or r.get("userId") or r.get("uid") or f"doc:{d.id}")
        uniq.add(uid)

    out["uniqueUsers"] = len(uniq)
    out["totalReports"] = total

    # Map counts to score
    if total == 0:
        s = 0.1
    elif total == 1:
        s = 0.2
    elif total <= 3:
        s = 0.5
    elif total <= 5:
        s = 0.7
    else:
        s = 0.9
    out["score"] = s
    return out


# ---------- Factor 2: Weather assist ----------
def _weather_factor(lat, lng):
    """
    Uses Open-Meteo current weather/precip. Returns 0.6 if raining/moderate, 0.4 if dry.
    """
    out = {"score": 0.5, "source": "open-meteo", "precip_mm": None, "isRaining": None}
    if lat is None or lng is None:
        return out
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat:.5f}&longitude={lng:.5f}&current=precipitation,weather_code"
        )
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        cur = (j.get("current") or {})
        precip = cur.get("precipitation")
        code = cur.get("weather_code")
        raining = (precip is not None and precip > 0) or (code in (51,53,55,61,63,65,66,67,80,81,82))

        if precip is not None:
            try:
                out["precip_mm"] = float(precip)
            except Exception:
                out["precip_mm"] = None

        out["isRaining"] = bool(raining)
        out["score"] = 0.6 if raining else 0.4  # doesn’t zero, per your note
        return out
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return out


# ---------- Factor 3: Hazard / history (your DB) ----------
def _hazard_context_factor(pin_lat, pin_lng):
    """
    Simple signals from your own DB:
      - recent verified reports in 24h within ~800m
      - active advisories in area (optional; collection 'advisories' with bbox/centroid)
    Returns score 0.4..0.9
    """
    base_score = 0.4
    if pin_lat is None or pin_lng is None:
        return {"score": base_score, "recentVerified": 0, "advisory": False}

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    minLat, maxLat, minLng, maxLng = _bbox(pin_lat, pin_lng, 800)

    # recent verified reports
    try:
        docs = db.collection("reports").where("createdAt", ">=", since).stream()
    except Exception:
        docs = []

    verified = 0
    for d in docs:
        r = d.to_dict() or {}
        lat, lng = _extract_lat_lng(r)
        if lat is None or lng is None:
            continue
        if not (minLat <= lat <= maxLat and minLng <= lng <= maxLng):
            continue
        v = ((r.get("aiVerification") or {}).get("status") or "").lower()
        if v == "verified":
            verified += 1

    advisory = False
    try:
        # OPTIONAL: your small 'advisories' collection
        q = db.collection("advisories").where("active", "==", True).limit(50).stream()
        for a in q:
            ad = a.to_dict() or {}
            alat, alng = _extract_lat_lng(ad)
            if alat is None or alng is None:
                continue
            if _haversine_m(pin_lat, pin_lng, alat, alng) <= 2000:
                advisory = True
                break
    except Exception:
        pass

    score = base_score
    if verified >= 20: score = 0.9
    elif verified >= 10: score = 0.75
    elif verified >= 3: score = 0.6
    else: score = 0.45

    if advisory:
        score = min(0.95, score + 0.1)

    return {"score": score, "recentVerified": verified, "advisory": advisory}


# ---------- Weighted final score ----------
WEIGHTS = {
    "visionFlood":   0.35,
    "depthMatch":    0.10,
    "exifFreshness": 0.10,
    "aiGenRisk":     0.10,
    "localCluster":  0.25,
    "weatherAssist": 0.05,
    "hazardContext": 0.05,
}

def _combine_scores(factors):
    def _g(v, default=0.5):  # gentle default
        try:
            x = float(v)
            return max(0.0, min(1.0, x))
        except Exception:
            return default

    # depthMatch: map label to score
    dm = factors["depthMatch"]["label"]
    if dm == "match": dm_score = 1.0
    elif dm == "near": dm_score = 0.7
    elif dm == "mismatch": dm_score = 0.2
    else: dm_score = 0.5

    aigen_label = factors["aiGenRisk"]["label"]
    if aigen_label == "low": ai_score = 1.0
    elif aigen_label == "medium": ai_score = 0.5
    else: ai_score = 0.1

    parts = {
        "visionFlood":   _g(factors["visionFlood"]["score"]),
        "depthMatch":    dm_score,
        "exifFreshness": _g(factors["exifFreshness"]["score"]),
        "aiGenRisk":     ai_score,
        "localCluster":  _g(factors["localCluster"]["score"]),
        "weatherAssist": _g(factors["weatherAssist"]["score"]),
        "hazardContext": _g(factors["hazardContext"]["score"]),
    }

    finalScore = sum(parts[k]*WEIGHTS[k] for k in WEIGHTS)
    finalScore = max(0.0, min(1.0, finalScore))

    # Decision rule
    if aigen_label == "high":
        status = "suspicious"
    elif finalScore >= 0.65:
        status = "verified"
    elif finalScore < 0.35:
        status = "suspicious"
    else:
        status = "uncertain"

    return finalScore, status, parts


# ---------- Main flow ----------
@functions_framework.cloud_event
def verify_flood_image(cloud_event: CloudEvent):
    _ensure_clients()

    report_ref = None
    try:
        report_id = _report_id_from_subject(cloud_event)
        if not report_id:
            logger.error("No report ID found in event.")
            return

        logger.info(f"Processing report: {report_id}")

        report_ref = db.collection("reports").document(report_id)
        snap = report_ref.get()
        if not snap.exists:
            logger.error(f"Report {report_id} not found.")
            return
        report_data = snap.to_dict() or {}

        media = report_data.get("media")
        if not isinstance(media, list) or not media:
            logger.warning("No media found in report.")
            return
        image_url = (media[0] or {}).get("url") or ""
        if not image_url:
            logger.warning("No image URL found.")
            return

        # User-provided depth (optional, safe parse)
        raw_depth = report_data.get("depth") or report_data.get("severity")
        user_depth = str(raw_depth).lower() if raw_depth is not None else None

        # Location & upload time
        pin_lat, pin_lng = _extract_lat_lng(report_data)
        upload_dt = _ts_to_dt(report_data.get("createdAt")) or datetime.now(timezone.utc)

        # Download + normalize image (keep raw for EXIF)
        for _ in range(2):  # tiny retry
            try:
                r = requests.get(image_url, timeout=30)
                r.raise_for_status()
                break
            except Exception:
                if _ == 1:
                    raise
        raw_bytes = r.content
        pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pil.thumbnail((2048, 2048))

        # ---- 1 & 5 & 8: Vision + Depth + AI-gen
        vis = _vision_flood_depth_aigen(pil, user_depth)

        # ---- 4: EXIF freshness
        exif_res = _parse_exif_freshness(Image.open(io.BytesIO(raw_bytes)), pin_lat, pin_lng, upload_dt)

        # ---- 6: Local cluster (density)
        cluster = _cluster_factor(pin_lat, pin_lng, window_minutes=90, radius_m=500)

        # ---- 2: Weather assist
        weather = _weather_factor(pin_lat, pin_lng)

        # ---- 3: Hazard/history context
        hazard = _hazard_context_factor(pin_lat, pin_lng)

        # Build factor objects for combiner
        factors = {
            "visionFlood": {
                "score": vis["visionConfidence"] if vis["isFloodWater"] else (1.0 - vis["visionConfidence"]),
                "isFloodWater": vis["isFloodWater"],
                "modelDepth": vis["depthLabel"],
                "reasoning": vis["reasoning"],
            },
            "depthMatch": {
                "label": vis["depthAgreement"],  # match | near | mismatch | unknown
                "userDepth": user_depth or "unknown",
                "modelDepth": vis["depthLabel"],
            },
            "exifFreshness": exif_res,
            "aiGenRisk": {
                "label": vis["aiGenLabel"],
                "confidence": vis["aiGenConfidence"],
            },
            "localCluster": cluster,
            "weatherAssist": weather,
            "hazardContext": hazard,
        }

        finalScore, status, parts_used = _combine_scores(factors)

        update = {
            "aiVerification": {
                "status": status,
                "finalScore": float(finalScore),
                "verifiedAt": firestore.SERVER_TIMESTAMP,
                "factors": factors,
                "meta": {
                    "weights": WEIGHTS,
                    "partsUsed": parts_used,
                }
            }
        }

        # Back-compat minimal fields
        update["aiVerification"]["isFloodWater"] = bool(vis["isFloodWater"])
        update["aiVerification"]["confidence"] = float(vis["visionConfidence"])

        report_ref.update(update)
        logger.info(f"Report {report_id} scored: status={status} score={finalScore:.2f}")
        logger.info(f"Scored parts: {parts_used}")

    except Exception as e:
        logger.error(f"Error verifying image: {e}")
        try:
            if report_ref is not None:
                report_ref.update({
                    "aiVerification": {
                        "error": str(e),
                        "status": "failed",
                        "verifiedAt": firestore.SERVER_TIMESTAMP,
                    }
                })
        except Exception as ee:
            logger.error(f"Failed to write failure status: {ee}")


# ---------- Local test runner ----------
if __name__ == "__main__":
    from functions_framework import create_app
    app = create_app(target="verify_flood_image", signature_type="cloudevent")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
