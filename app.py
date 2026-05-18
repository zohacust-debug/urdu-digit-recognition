from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

import numpy as np
import cv2
from PIL import Image
import io, base64, os


# ------------------- APP CONFIG -------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MODEL_PATH = os.environ.get("URDU_MODEL_PATH", r"D:\urdu_digit_recognition\uploads\urdu_cnn_model.h5")
model = load_model(MODEL_PATH, compile=False)

TARGET_SIZE = 128


# ------------------- HELPERS -------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_b64_png(img_bgr_or_gray: np.ndarray) -> str | None:
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


# ------------------- CORE PREPROCESS (MATCH YOUR DATASET) -------------------
# Your dataset is: WHITE digit on BLACK background (often near-binary)
# So inference must produce: background=0, digit=255 (then /255 => 0..1)

def normalize_and_reshape(gray_128: np.ndarray) -> np.ndarray:
    x = gray_128.astype("float32") / 255.0
    return x.reshape(1, TARGET_SIZE, TARGET_SIZE, 1)


def auto_to_white_on_black(gray: np.ndarray) -> np.ndarray:
    """
    Ensures digit becomes bright (white) and background becomes dark (black).
    If image background is white (mean high), invert.
    """
    if gray is None:
        return gray
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)
    return gray


def crop_center_white_on_black(gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns:
      - processed_128 (uint8) [128x128], white digit on black background
      - bw_mask (uint8) for debugging/preview (also white digit on black)
    """
    if gray is None:
        return None, None

    gray = auto_to_white_on_black(gray)

    # Slight blur helps threshold stability
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold: digit should be white
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(bw)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)

    # Crop from bw (we want near-binary white-on-black)
    digit = bw[y:y+h, x:x+w]

    # Square black canvas + padding
    side = max(w, h) + 16
    canvas = np.zeros((side, side), dtype=np.uint8)
    xoff = (side - w) // 2
    yoff = (side - h) // 2
    canvas[yoff:yoff+h, xoff:xoff+w] = digit

    # Resize to model input
    out = cv2.resize(canvas, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    return out, bw


def preprocess_single_from_path(path: str) -> tuple[np.ndarray | None, str | None]:
    """
    Returns:
      - model_input (1,128,128,1)
      - preview_b64 (processed image)
    """
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed_128, _ = crop_center_white_on_black(gray)
    if processed_128 is None:
        return None, None
    return normalize_and_reshape(processed_128), to_b64_png(processed_128)


def preprocess_single_from_canvas(pil_img: Image.Image) -> tuple[np.ndarray | None, str | None]:
    gray = np.array(pil_img.convert("L"))
    processed_128, _ = crop_center_white_on_black(gray)
    if processed_128 is None:
        return None, None
    return normalize_and_reshape(processed_128), to_b64_png(processed_128)


# ------------------- MULTI-DIGIT (FAST SEGMENT + BATCH PREDICT) -------------------
def segment_digits(gray: np.ndarray) -> tuple[list[tuple[int,int,int,int]], np.ndarray | None]:
    """
    Fast segmentation using connected components.
    Returns boxes sorted left->right and bw mask (white digit on black).
    """
    if gray is None:
        return [], None

    gray = auto_to_white_on_black(gray)
    H, W = gray.shape

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,  # digit should be white after inversion
        41, 7
    )

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((4,4), np.uint8), iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    boxes = []
    min_area = 0.002 * H * W
    max_area = 0.35 * H * W
    margin = 3

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if x < margin or y < margin or x+w > W-margin or y+h > H-margin:
            continue
        if area < min_area or area > max_area:
            continue
        if w < 8 or h < 18:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    # Simple split for very wide components (two digits stuck)
    final_boxes = []
    for (x, y, w, h) in boxes:
        if w > 1.25 * h:
            roi = bw[y:y+h, x:x+w]
            col_sum = np.sum(roi > 0, axis=0).astype(np.float32)
            mid = w // 2
            left = max(0, mid - w//4)
            right = min(w, mid + w//4)
            cut = left + int(np.argmin(col_sum[left:right]))
            if 6 < cut < w - 6:
                final_boxes.append((x, y, cut, h))
                final_boxes.append((x + cut, y, w - cut, h))
            else:
                final_boxes.append((x, y, w, h))
        else:
            final_boxes.append((x, y, w, h))

    final_boxes.sort(key=lambda b: b[0])
    return final_boxes, bw


def crop_box_to_128(bw: np.ndarray, box: tuple[int,int,int,int]) -> np.ndarray:
    """
    bw is already white digit on black.
    Return 128x128 (uint8) also white on black.
    """
    x, y, w, h = box
    H, W = bw.shape

    pad = int(0.18 * max(w, h)) + 6
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + pad, W), min(y + h + pad, H)

    crop = bw[y1:y2, x1:x2]
    ch, cw = crop.shape

    side = max(ch, cw) + 8
    canvas = np.zeros((side, side), dtype=np.uint8)
    xoff = (side - cw) // 2
    yoff = (side - ch) // 2
    canvas[yoff:yoff+ch, xoff:xoff+cw] = crop

    out = cv2.resize(canvas, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    return out


def predict_multi_digit(path: str, return_preview=True):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    boxes, bw = segment_digits(gray)

    if gray is None or bw is None or not boxes:
        return "", [], None

    # Build batch
    batch_imgs = []
    clean_boxes = []

    for b in boxes:
        img128 = crop_box_to_128(bw, b)
        batch_imgs.append(img128)
        clean_boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])

    batch = np.stack(batch_imgs, axis=0).astype("float32") / 255.0
    batch = batch.reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)

    probs_all = model.predict(batch, verbose=0)  # (N,10)
    pred_ids = np.argmax(probs_all, axis=1).astype(int)
    confs = np.max(probs_all, axis=1).astype(float)

    digits = [str(int(d)) for d in pred_ids]
    details = []
    for box, d, c in zip(clean_boxes, pred_ids, confs):
        details.append({
            "box": box,
            "digit": int(d),
            "confidence": float(round(c, 4))
        })

    preview_b64 = None
    if return_preview:
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for item in details:
            x, y, w, h = item["box"]
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                preview,
                f"{item['digit']} {item['confidence']:.2f}",
                (x, max(0, y-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        preview_b64 = to_b64_png(preview)

    return "".join(digits), details, preview_b64


# ------------------- ROUTES -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    preview = None
    error = None

    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if not file or not file.filename:
            error = "کوئی فائل منتخب نہیں ہوئی"
        elif not allowed_file(file.filename):
            error = "صرف PNG/JPG/JPEG فائل اپلوڈ کریں"
        else:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            x_in, preview = preprocess_single_from_path(path)
            if x_in is None:
                error = "ہندسہ نہیں ملا، براہ کرم واضح تصویر اپلوڈ کریں"
            else:
                probs = model.predict(x_in, verbose=0)[0]
                prediction = int(np.argmax(probs))
                confidence = float(np.max(probs))

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           preview=preview,
                           error=error)


@app.route("/canvas", methods=["POST"])
def canvas_predict():
    data = request.json.get("image_data")
    if not data:
        return jsonify({"error": "کوئی ہندسہ نہیں ملا"})

    img_bytes = base64.b64decode(data.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes))

    x_in, preview = preprocess_single_from_canvas(img)
    if x_in is None:
        return jsonify({"error": "ہندسہ نہیں ملا"})

    probs = model.predict(x_in, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return jsonify({"prediction": pred, "confidence": round(conf, 4), "preview": preview})


@app.route("/multi_digit", methods=["POST"])
def multi_digit():
    if "file" not in request.files:
        return jsonify({"error": "کوئی فائل نہیں ملی"})

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "کوئی فائل منتخب نہیں ہوئی"})

    if not allowed_file(file.filename):
        return jsonify({"error": "صرف PNG/JPG/JPEG فائل اپلوڈ کریں"})

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    text, details, preview = predict_multi_digit(path, return_preview=True)
    if not text:
        return jsonify({"prediction": "", "details": [], "preview": None, "error": "کوئی ہندسہ نہیں ملا"})

    return jsonify({"prediction": text, "details": details, "preview": preview})


if __name__ == "__main__":
    app.run(debug=True)
