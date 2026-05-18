// Sidebar navigation
const sidebarItems = document.querySelectorAll(".sidebar ul li");
const modules = document.querySelectorAll(".module");

sidebarItems.forEach((item) => {
  item.addEventListener("click", () => {
    sidebarItems.forEach((i) => i.classList.remove("active"));
    item.classList.add("active");

    const moduleId = item.getAttribute("data-module");
    modules.forEach((m) => {
      m.id === moduleId ? m.classList.add("active") : m.classList.remove("active");
    });
  });
});

// Canvas
const canvas = document.getElementById("draw-canvas");
if (canvas) {
  const ctx = canvas.getContext("2d");
  let drawing = false;

  // White canvas (user draws black)
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.strokeStyle = "black";

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    if (e.touches && e.touches[0]) {
      return { x: e.touches[0].clientX - rect.left, y: e.touches[0].clientY - rect.top };
    }
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function startDraw(e) {
    drawing = true;
    const p = getPos(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  }

  function endDraw() {
    drawing = false;
    ctx.beginPath();
  }

  function draw(e) {
    if (!drawing) return;
    e.preventDefault();
    const p = getPos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mouseup", endDraw);
  canvas.addEventListener("mouseleave", endDraw);
  canvas.addEventListener("mousemove", draw);

  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchend", endDraw);
  canvas.addEventListener("touchcancel", endDraw);
  canvas.addEventListener("touchmove", draw, { passive: false });

  const canvasResult = document.getElementById("canvas-result");
  const canvasPreviewWrap = document.getElementById("canvas-preview-wrap");
  const canvasPreviewImg = document.getElementById("canvas-preview");

  document.getElementById("clear-btn").onclick = () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    canvasResult.innerHTML = "";
    canvasPreviewWrap.classList.add("hidden");
    canvasPreviewImg.src = "";
  };

  document.getElementById("predict-btn").onclick = async () => {
    canvasResult.innerHTML = `<p class="muted">پیشگوئی ہو رہی ہے...</p>`;
    const dataURL = canvas.toDataURL("image/png");

    try {
      const res = await fetch("/canvas", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: dataURL }),
      });

      const data = await res.json();

      if (data.error) {
        canvasResult.innerHTML = `<h2>${data.error}</h2>`;
        canvasPreviewWrap.classList.add("hidden");
        return;
      }

      const conf = data.confidence !== undefined ? ` (Confidence: ${(data.confidence * 100).toFixed(2)}%)` : "";
      canvasResult.innerHTML = `<h2>پیشگوئی شدہ ہندسہ: ${data.prediction}${conf}</h2>`;

      if (data.preview) {
        canvasPreviewWrap.classList.remove("hidden");
        canvasPreviewImg.src = data.preview;
      }
    } catch (e) {
      console.error(e);
      canvasResult.innerHTML = `<h2>سرور ایرر</h2>`;
    }
  };
}

// Multi-digit AJAX
const multiDigitForm = document.getElementById("multi-digit-form");
if (multiDigitForm) {
  const resultDiv = document.getElementById("multi-digit-result");
  const previewWrap = document.getElementById("multi-preview-wrap");
  const previewImg = document.getElementById("multi-preview");
  const detailsWrap = document.getElementById("multi-details-wrap");

  multiDigitForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultDiv.innerHTML = `<p class="muted">پیشگوئی ہو رہی ہے...</p>`;
    detailsWrap.innerHTML = "";
    previewWrap.classList.add("hidden");
    previewImg.src = "";

    const formData = new FormData(multiDigitForm);

    try {
      const res = await fetch("/multi_digit", { method: "POST", body: formData });
      const data = await res.json();

      if (data.error && !data.prediction) {
        resultDiv.innerHTML = `<h2>${data.error}</h2>`;
        return;
      }

      resultDiv.innerHTML = `<h2>پیشگوئی شدہ متن: ${data.prediction || "—"}</h2>`;

      if (data.preview) {
        previewWrap.classList.remove("hidden");
        previewImg.src = data.preview;
      }

      if (Array.isArray(data.details) && data.details.length) {
        const rows = data.details
          .map((d, i) => {
            const c = (d.confidence * 100).toFixed(2);
            const b = d.box.join(", ");
            return `<tr><td>${i + 1}</td><td>${d.digit}</td><td>${c}%</td><td>${b}</td></tr>`;
          })
          .join("");

        detailsWrap.innerHTML = `
          <h3>Details</h3>
          <div class="table-wrap">
            <table>
              <thead><tr><th>#</th><th>Digit</th><th>Confidence</th><th>Box (x,y,w,h)</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        `;
      }
    } catch (e) {
      console.error(e);
      resultDiv.innerHTML = `<h2>سرور ایرر</h2>`;
    }
  });
}
