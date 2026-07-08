// Modern Asynchronous OCR Frontend Controller
const API_URL = ""; // Relative path to FastAPI backend

let currentMethod = "knn";
let uploadedFileBytes = null;
let currentPredictions = [];
let selectedIndex = 0;
let statusData = null;

// DOM Cache
const dropzone = document.getElementById("ocr-dropzone");
const fileInput = document.getElementById("ocr-file-input");
const dropzonePrompt = document.getElementById("dropzone-prompt");
const canvasContainer = document.getElementById("canvas-container");
const canvas = document.getElementById("ocr-canvas");
const ctx = canvas.getContext("2d");
const runOcrBtn = document.getElementById("run-ocr-btn");
const clearBtn = document.getElementById("clear-btn");

const emptyResults = document.getElementById("empty-results-state");
const activeResults = document.getElementById("active-results-state");
const predOutputString = document.getElementById("pred-output-string");
const charTilesWrapper = document.getElementById("char-tiles-wrapper");
const momentsBox = document.getElementById("moments-box");

// Scale factors for plotting bounding boxes on coordinate space
let scaleX = 1.0;
let scaleY = 1.0;
let originalImage = null;

window.addEventListener("DOMContentLoaded", () => {
    fetchSystemStatus();
    setupDropzone();
});

// Fetch system status
async function fetchSystemStatus() {
    try {
        const response = await fetch(`${API_URL}/api/status`);
        const status = await response.json();
        statusData = status;

        // Render Model Badges
        updateBadge("status-knn", status.knn_trained);
        updateBadge("status-cnn", status.cnn_trained && status.has_tf);

        // Populate dropdown
        const select = document.getElementById("template-class-select");
        select.innerHTML = '<option value="" disabled selected>Select class character</option>';
        status.allowed_classes.forEach(c => {
            const opt = document.createElement("option");
            opt.value = c;
            opt.textContent = `Character '${c}'`;
            select.appendChild(opt);
        });

        // Render Distribution
        const grid = document.getElementById("distribution-grid");
        grid.innerHTML = "";
        status.allowed_classes.forEach(c => {
            const count = status.class_counts[c] || 0;
            const card = document.createElement("div");
            card.className = "dist-card";
            card.innerHTML = `
                <span class="l-lbl">${c.toUpperCase()}</span>
                <span class="l-count">${count} items</span>
            `;
            grid.appendChild(card);
        });

    } catch (e) {
        showToast("Error connecting to server. Make sure FastAPI server is running.");
        console.error(e);
    }
}

function updateBadge(elementId, isTrained) {
    const badge = document.getElementById(elementId);
    const valueSpan = badge.querySelector(".badge-val");
    if (isTrained) {
        badge.className = "status-badge trained";
        valueSpan.textContent = "Trained";
    } else {
        badge.className = "status-badge";
        valueSpan.textContent = "Untrained";
    }
}

function setMethod(method) {
    currentMethod = method;
    document.getElementById("method-knn").classList.toggle("active", method === "knn");
    document.getElementById("method-cnn").classList.toggle("active", method === "cnn");
    if (uploadedFileBytes) {
        triggerPrediction();
    }
}

// Drag & drop handlers
function setupDropzone() {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('hover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('hover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

function handleFileUpload(file) {
    uploadedFileBytes = file;
    const reader = new FileReader();
    reader.onload = (event) => {
        originalImage = new Image();
        originalImage.onload = () => {
            renderImageOnCanvas();
            dropzonePrompt.style.display = "none";
            canvasContainer.style.display = "flex";
            runOcrBtn.style.display = "inline-flex";
            clearBtn.style.display = "inline-flex";
        };
        originalImage.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function renderImageOnCanvas(activeHoverIndex = null) {
    if (!originalImage) return;

    // Standard scale bounding box logic to display scaled image without stretch
    const maxW = 500;
    const maxH = 400;
    let canvasW = originalImage.width;
    let canvasH = originalImage.height;

    if (canvasW > maxW) {
        canvasH = Math.round(canvasH * (maxW / canvasW));
        canvasW = maxW;
    }
    if (canvasH > maxH) {
        canvasW = Math.round(canvasW * (maxH / canvasH));
        canvasH = maxH;
    }

    canvas.width = canvasW;
    canvas.height = canvasH;

    scaleX = canvasW / originalImage.width;
    scaleY = canvasH / originalImage.height;

    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.drawImage(originalImage, 0, 0, canvasW, canvasH);

    // Draw bounding boxes on top of canvas
    currentPredictions.forEach(pred => {
        const x = pred.bbox.x * scaleX;
        const y = pred.bbox.y * scaleY;
        const w = pred.bbox.w * scaleX;
        const h = pred.bbox.h * scaleY;

        const isHovered = (activeHoverIndex === pred.index);
        const isActive = (selectedIndex === pred.index);

        if (isActive) {
            ctx.strokeStyle = "#00f2a1"; // Green active
            ctx.lineWidth = 3;
        } else if (isHovered) {
            ctx.strokeStyle = "#4facfe"; // Cyan hover
            ctx.lineWidth = 2;
        } else {
            ctx.strokeStyle = "#00f2fe"; // Alpha cyan normal
            ctx.lineWidth = 1.5;
        }
        ctx.strokeRect(x, y, w, h);

        // Write small class text target
        if (isActive || isHovered) {
            ctx.fillStyle = isActive ? "#00f2a1" : "#4facfe";
            ctx.font = "bold 12px Space Grotesk";
            ctx.fillText(pred.label.toUpperCase(), x, y - 4);
        }
    });
}

// Canvas cursor interactives
canvas.addEventListener("mousemove", (e) => {
    if (currentPredictions.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let hoverIndex = null;
    for (const pred of currentPredictions) {
        const x = pred.bbox.x * scaleX;
        const y = pred.bbox.y * scaleY;
        const w = pred.bbox.w * scaleX;
        const h = pred.bbox.h * scaleY;

        if (mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + h) {
            hoverIndex = pred.index;
            break;
        }
    }
    renderImageOnCanvas(hoverIndex);
});

canvas.addEventListener("click", (e) => {
    if (currentPredictions.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    for (const pred of currentPredictions) {
        const x = pred.bbox.x * scaleX;
        const y = pred.bbox.y * scaleY;
        const w = pred.bbox.w * scaleX;
        const h = pred.bbox.h * scaleY;

        if (mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + h) {
            showCharacterDetails(pred.index);
            break;
        }
    }
});

// OCR Trigger action
async function triggerPrediction() {
    if (!uploadedFileBytes) return;

    // Toast loader anim
    showToast(`Evaluating character shapes using ${currentMethod.toUpperCase()}...`);

    const formData = new FormData();
    formData.append("file", uploadedFileBytes);

    try {
        const response = await fetch(`${API_URL}/api/predict?method=${currentMethod}`, {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            showToast(`Server: ${data.error}`);
            return;
        }

        currentPredictions = data.predictions;
        predOutputString.textContent = data.text || "Empty / Unrecognized";

        renderImageOnCanvas();
        populateCharacterTiles();

        emptyResults.style.display = "none";
        activeResults.style.display = "flex";

        if (currentPredictions.length > 0) {
            showCharacterDetails(1);
        } else {
            momentsBox.style.display = "none";
        }
    } catch (e) {
        showToast("Error running model prediction.");
        console.error(e);
    }
}

// Populate grid card list for extracted tiles
function populateCharacterTiles() {
    charTilesWrapper.innerHTML = "";
    currentPredictions.forEach(pred => {
        const tile = document.createElement("div");
        tile.className = `char-tile ${pred.index === selectedIndex ? 'active' : ''}`;
        tile.onclick = () => showCharacterDetails(pred.index);
        tile.innerHTML = `
            <img src="${pred.roi_image}" alt="roi">
            <span class="tile-badge">${pred.label.toUpperCase()}</span>
        `;
        charTilesWrapper.appendChild(tile);
    });
}

function showCharacterDetails(index) {
    selectedIndex = index;
    populateCharacterTiles();
    renderImageOnCanvas();

    const pred = currentPredictions.find(p => p.index === index);
    if (!pred) return;

    document.getElementById("selected-char-name").textContent = `Component #${index} Info`;

    const container = document.getElementById("moments-numeric-values");
    container.innerHTML = "";

    if (pred.hu_moments && pred.hu_moments.length > 0) {
        momentsBox.style.display = "flex";
        pred.hu_moments.forEach((val, idx) => {
            const block = document.createElement("div");
            block.className = "moment-cell";
            block.innerHTML = `
                <span class="m-lbl">Hu ${idx + 1}</span>
                <span class="m-val">${val.toExponential(3)}</span>
            `;
            container.appendChild(block);
        });
    } else {
        // Fallback for CNN (CNN outputs prediction probabilities)
        momentsBox.style.display = "flex";
        container.innerHTML = `
            <div class="moment-cell" style="grid-column: span 4; padding: 12px;">
                <span class="m-lbl">CNN Classification Confidence</span>
                <span class="m-val" style="font-size: 1.1rem; color: #00f2a1; font-weight: 800;">
                    ${(pred.prob * 100).toFixed(2)}%
                </span>
            </div>
        `;
    }
}

// Upload character grids (training bitmaps)
async function uploadTemplate(e) {
    e.preventDefault();
    const select = document.getElementById("template-class-select");
    const fileIn = document.getElementById("template-file");
    const char = select.value;
    const file = fileIn.files[0];

    if (!char || !file) {
        showToast("Select target class label and insert BMP grid file.");
        return;
    }

    const formData = new FormData();
    formData.append("label_char", char);
    formData.append("file", file);

    const btn = document.getElementById("upload-template-btn");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing...`;

    try {
        const response = await fetch(`${API_URL}/api/upload-template`, {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (response.ok) {
            showToast(data.message);
            fileIn.value = "";
            fetchSystemStatus();
        } else {
            showToast(`Upload Error: ${data.detail || "Server error"}`);
        }
    } catch (err) {
        showToast("Error uploading file to server.");
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-file-export"></i> Segment & Add to Dataset`;
    }
}

// Trigger Async Learning
async function triggerTraining() {
    const btn = document.getElementById("train-models-btn");
    const spinner = document.getElementById("training-spinner");

    btn.style.display = "none";
    spinner.style.display = "flex";

    try {
        const response = await fetch(`${API_URL}/api/train`, {
            method: "POST"
        });
        const data = await response.json();

        if (data.success) {
            let msg = "K-NN learning moments matrix successfully trained!";
            if (data.cnn_success) {
                msg += " Keras CNN model trained!";
            } else {
                msg += ` (CNN: ${data.cnn_message})`;
            }
            showToast(msg);
            fetchSystemStatus();
        } else {
            showToast(data.message || "Model training failed.");
        }
    } catch (e) {
        showToast("Error processing standard training fits.");
        console.error(e);
    } finally {
        btn.style.display = "inline-flex";
        spinner.style.display = "none";
    }
}

// Reset view interface
function resetPlayground() {
    uploadedFileBytes = null;
    currentPredictions = [];
    selectedIndex = 0;
    originalImage = null;

    fileInput.value = "";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvasContainer.style.display = "none";
    dropzonePrompt.style.display = "flex";
    runOcrBtn.style.display = "none";
    clearBtn.style.display = "none";

    emptyResults.style.display = "flex";
    activeResults.style.display = "none";
    momentsBox.style.display = "none";
}

// Toast helper layout
function showToast(message) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = "toast show";
    setTimeout(() => {
        toast.className = "toast";
    }, 4500);
}
