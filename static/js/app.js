const input = document.querySelector("#image-input");
const uploadZone = document.querySelector("#upload-zone");
const fileList = document.querySelector("#file-list");
const form = document.querySelector("#analysis-form");
const submitButton = form.querySelector("button[type='submit']");
let selectedFiles = [];
let probabilityChart;
let scoreChart;
let latestReport;

const classColors = { Healthy: "#70b846", Stressed: "#c55f4b" };

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem("canopy-health-history") || "[]");
  } catch (error) {
    return [];
  }
}

function setHistory(entries) {
  localStorage.setItem("canopy-health-history", JSON.stringify(entries.slice(0, 5)));
}

function renderHistory() {
  const history = getHistory();
  const list = document.querySelector("#history-list");
  if (!list) return;
  list.replaceChildren();
  if (!history.length) {
    const empty = document.createElement("li");
    empty.className = "history-empty";
    empty.textContent = "No recent analyses yet.";
    list.append(empty);
    return;
  }
  history.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "history-item";
    const label = document.createElement("span");
    label.textContent = entry.label;
    const detail = document.createElement("small");
    detail.textContent = `${entry.category} · ${entry.score} / 100`;
    item.append(label, detail);
    list.append(item);
  });
}

function updateRecommendation(recommendation) {
  const title = document.querySelector("#care-title");
  const message = document.querySelector("#care-message");
  if (!recommendation || !title || !message) return;
  title.textContent = recommendation.title || "Maintain current care";
  message.textContent = recommendation.message || "Continue routine monitoring.";
}

function setFiles(files) {
  selectedFiles = [...selectedFiles, ...files].filter((file) => file.type.startsWith("image/")).slice(0, 12);
  renderFileList();
}

function renderFileList() {
  fileList.replaceChildren();
  selectedFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-row";
    const preview = document.createElement("img");
    preview.src = URL.createObjectURL(file);
    preview.alt = "Selected image preview";
    const fields = document.createElement("div");
    fields.className = "file-fields";
    const viewName = document.createElement("input");
    viewName.value = `View ${index + 1}`;
    viewName.setAttribute("aria-label", `Name for image ${index + 1}`);
    viewName.dataset.view = "true";
    const filename = document.createElement("input");
    filename.value = file.name;
    filename.readOnly = true;
    filename.title = file.name;
    fields.append(viewName, filename);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove-file";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.addEventListener("click", () => {
      selectedFiles.splice(index, 1);
      renderFileList();
    });
    row.append(preview, fields, remove);
    fileList.append(row);
  });
}

input.addEventListener("change", () => {
  setFiles([...input.files]);
  input.value = "";
});
for (const eventName of ["dragenter", "dragover"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragover");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("dragover");
  });
}
uploadZone.addEventListener("drop", (event) => setFiles([...event.dataTransfer.files]));
document.querySelectorAll(".mode-option input").forEach((radio) => {
  radio.addEventListener("change", () => {
    document.querySelectorAll(".mode-option").forEach((option) => option.classList.remove("selected"));
    radio.closest(".mode-option").classList.add("selected");
  });
});

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function updateFusion(method) {
  if (!latestReport) return;
  const fused = latestReport.fusion[method];
  const category = fused.health_category;
  const confidence = percent(fused.confidence);
  const finalCategory = document.querySelector("#final-category");
  const finalConfidence = document.querySelector("#final-confidence");
  const finalScore = document.querySelector("#final-score");
  const scoreRing = document.querySelector("#score-ring");
  const predictionPill = document.querySelector("#prediction-pill");
  const predictionConfidence = document.querySelector("#prediction-confidence");
  const predictionScore = document.querySelector("#prediction-score");
  const summaryNote = document.querySelector("#summary-note");
  if (finalCategory) finalCategory.textContent = category;
  if (finalConfidence) finalConfidence.textContent = `${confidence} fused confidence · ${method.replaceAll("_", " ")}`;
  if (finalScore) finalScore.textContent = fused.health_score.toFixed(0);
  if (scoreRing) scoreRing.style.setProperty("--score", fused.health_score);
  if (predictionPill) {
    predictionPill.textContent = category;
    predictionPill.className = `prediction-pill ${category.toLowerCase().replaceAll(" ", "-")}`;
  }
  if (predictionConfidence) predictionConfidence.textContent = confidence;
  if (predictionScore) predictionScore.textContent = `${fused.health_score.toFixed(0)} / 100`;
  if (summaryNote) summaryNote.textContent = `${method.replaceAll("_", " ")} result across ${latestReport.images.length} view${latestReport.images.length === 1 ? "" : "s"}.`;
  document.querySelectorAll(".fusion-card").forEach((card) => card.classList.toggle("active", card.dataset.method === method));
  if (!probabilityChart) return;
  const probabilities = fused.probabilities;
  probabilityChart.data.datasets[0].data = Object.values(probabilities).map((value) => value * 100);
  probabilityChart.update();
}

function buildCharts(report) {
  if (probabilityChart) probabilityChart.destroy();
  if (scoreChart) scoreChart.destroy();
  probabilityChart = new Chart(document.querySelector("#probability-chart"), {
    type: "bar",
    data: {
      labels: Object.keys(report.fusion.confidence_weighted.probabilities),
      datasets: [{ data: Object.keys(report.fusion.confidence_weighted.probabilities).map(() => 0), backgroundColor: Object.keys(report.fusion.confidence_weighted.probabilities).map((name) => classColors[name]), borderRadius: 2, maxBarThickness: 34 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (context) => `${context.raw.toFixed(1)}%` } } },
      scales: { y: { beginAtZero: true, max: 100, ticks: { stepSize: 25, callback: (value) => `${value}%`, font: { size: 9 } }, grid: { color: "#edf0eb" } }, x: { grid: { display: false }, ticks: { font: { size: 9 } } } },
    },
  });
  scoreChart = new Chart(document.querySelector("#score-chart"), {
    type: "line",
    data: {
      labels: report.images.map((image) => image.view),
      datasets: [{ data: report.images.map((image) => image.health_score), borderColor: "#236a4b", backgroundColor: "#dfeadd", fill: true, tension: 0.25, pointRadius: 4, pointBackgroundColor: "#70b846" }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (context) => `${context.raw.toFixed(1)} / 100` } } },
      scales: { y: { beginAtZero: true, max: 100, ticks: { stepSize: 25, font: { size: 9 } }, grid: { color: "#edf0eb" } }, x: { grid: { display: false }, ticks: { font: { size: 9 }, maxRotation: 0 } } },
    },
  });
  updateFusion(document.querySelector("#fusion-select").value);
}

function renderResults(report) {
  latestReport = report;
  const summary = report.summary || report.fusion.confidence_weighted;
  const recommendation = summary.recommendation || { title: "Maintain current care", message: "Continue routine monitoring." };
  const emptyState = document.querySelector("#empty-state");
  const results = document.querySelector("#results");
  const errorBanner = document.querySelector("#error-banner");
  const resultMode = document.querySelector("#result-mode");
  const reportLink = document.querySelector("#report-link");
  const viewCount = document.querySelector("#view-count");
  const finalCategory = document.querySelector("#final-category");
  const finalConfidence = document.querySelector("#final-confidence");
  const finalScore = document.querySelector("#final-score");
  const scoreRing = document.querySelector("#score-ring");
  const predictionPill = document.querySelector("#prediction-pill");
  const predictionConfidence = document.querySelector("#prediction-confidence");
  const predictionScore = document.querySelector("#prediction-score");
  const summaryNote = document.querySelector("#summary-note");
  if (emptyState) emptyState.hidden = true;
  if (results) results.hidden = false;
  if (errorBanner) errorBanner.hidden = true;
  if (resultMode) resultMode.textContent = report.mode === "demo" ? "DEMO MODE · HEURISTIC PREVIEW" : "TRAINED MODEL · MOBILENETV3";
  if (reportLink) reportLink.href = report.report_url;
  if (viewCount) viewCount.textContent = `${report.images.length} IMAGE${report.images.length === 1 ? "" : "S"}`;
  if (finalCategory) finalCategory.textContent = summary.health_category;
  if (finalConfidence) finalConfidence.textContent = `${percent(summary.confidence)} fused confidence`;
  if (finalScore) finalScore.textContent = summary.health_score.toFixed(0);
  if (scoreRing) scoreRing.style.setProperty("--score", summary.health_score);
  if (predictionPill) {
    predictionPill.textContent = summary.health_category;
    predictionPill.className = `prediction-pill ${summary.health_category.toLowerCase().replaceAll(" ", "-")}`;
  }
  if (predictionConfidence) predictionConfidence.textContent = percent(summary.confidence);
  if (predictionScore) predictionScore.textContent = `${summary.health_score.toFixed(0)} / 100`;
  if (summaryNote) summaryNote.textContent = `High-confidence fused estimate across ${report.images.length} views.`;
  updateRecommendation(recommendation);

  const history = getHistory();
  const entry = {
    label: `${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`,
    category: summary.health_category,
    score: summary.health_score.toFixed(0),
  };
  setHistory([entry, ...history].slice(0, 5));
  renderHistory();

  const cards = document.querySelector("#fusion-cards");
  if (cards) cards.replaceChildren();
  const methodLabels = { majority_vote: "Majority vote", average_probability: "Average probability", confidence_weighted: "Confidence-weighted" };
  for (const [method, result] of Object.entries(report.fusion)) {
    const card = document.createElement("div");
    card.className = "fusion-card";
    card.dataset.method = method;
    const tag = document.createElement("span");
    tag.textContent = methodLabels[method].toUpperCase();
    const category = document.createElement("strong");
    category.textContent = result.health_category;
    const score = document.createElement("small");
    score.textContent = `${result.health_score.toFixed(1)} / 100`;
    card.append(tag, category, score);
    cards.append(card);
  }

  const imageCards = document.querySelector("#image-cards");
  if (imageCards) imageCards.replaceChildren();
  for (const image of report.images) {
    const predictedClass = image.predicted_class || image.health_category || "Healthy";
    const probabilities = image.probabilities || {};
    const quality = image.quality || { sharpness: "—", brightness: "—", resolution: "—", vegetation_coverage: 0, flags: [] };
    const card = document.createElement("article");
    card.className = "image-card";
    const preview = document.createElement("img");
    preview.className = "image-preview";
    preview.src = image.image_url;
    preview.alt = `${image.view} original canopy image`;
    const body = document.createElement("div");
    body.className = "image-card-body";
    const titleRow = document.createElement("div");
    titleRow.className = "image-title-row";
    const title = document.createElement("h4");
    title.textContent = image.view;
    const badge = document.createElement("span");
    badge.className = `class-badge ${predictedClass.toLowerCase().replaceAll(" ", "-")}`;
    badge.textContent = predictedClass;
    titleRow.append(title, badge);
    const stats = document.createElement("div");
    stats.className = "image-stats";
    stats.textContent = `Confidence ${percent(image.confidence)} · Score ${image.health_score.toFixed(1)} / 100`;
    const probabilityList = document.createElement("div");
    probabilityList.className = "probability-list";
    for (const [className, value] of Object.entries(probabilities)) {
      const row = document.createElement("div");
      row.className = "probability-row";
      const label = document.createElement("span");
      label.textContent = className;
      const track = document.createElement("div");
      track.className = "probability-track";
      const fill = document.createElement("div");
      fill.className = "probability-fill";
      fill.style.width = percent(value);
      track.append(fill);
      const amount = document.createElement("span");
      amount.textContent = percent(value);
      row.append(label, track, amount);
      probabilityList.append(row);
    }
    const details = document.createElement("div");
    details.className = "image-stats";
    details.textContent = `Sharpness ${quality.sharpness} · Brightness ${quality.brightness} · ${quality.resolution} px · Vegetation ${(quality.vegetation_coverage * 100).toFixed(1)}%`;
    body.append(titleRow, stats, probabilityList, details);
    if (Array.isArray(quality.flags) && quality.flags.length) {
      const flags = document.createElement("div");
      flags.className = "quality-flags";
      flags.textContent = `Quality flags: ${quality.flags.join(", ")}`;
      body.append(flags);
    }
    if (image.gradcam_url) {
      const pair = document.createElement("div");
      pair.className = "gradcam-pair";
      for (const [src, labelText] of [[image.image_url, "Original"], [image.gradcam_url, "Grad-CAM overlay"]]) {
        const wrapper = document.createElement("div");
        const img = document.createElement("img");
        img.src = src;
        img.alt = `${image.view} ${labelText}`;
        const caption = document.createElement("span");
        caption.textContent = labelText;
        wrapper.append(img, caption);
        pair.append(wrapper);
      }
      body.append(pair);
    }
    card.append(preview, body);
    imageCards.append(card);
  }
  buildCharts(report);
}

document.querySelector("#fusion-select").addEventListener("change", (event) => updateFusion(event.target.value));
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const errorBanner = document.querySelector("#error-banner");
  if (!selectedFiles.length) {
    errorBanner.textContent = "Choose at least one canopy image first.";
    errorBanner.hidden = false;
    return;
  }
  const data = new FormData();
  const mode = form.querySelector("input[name='mode']:checked").value;
  data.append("mode", mode);
  data.append("quality_weighted", document.querySelector("#quality-weighting").checked ? "true" : "false");
  selectedFiles.forEach((file, index) => {
    data.append("images", file);
    data.append(`view_${index + 1}`, fileList.querySelectorAll("input[data-view]")[index].value || `View ${index + 1}`);
  });
  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Analyzing views…";
  document.querySelector("#results").hidden = true;
  document.querySelector("#empty-state").hidden = false;
  errorBanner.hidden = true;
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Analysis failed.");
    renderResults(result);
  } catch (error) {
    errorBanner.textContent = error.message || "Could not reach the analysis service.";
    errorBanner.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Analyze image set";
  }
});

renderHistory();