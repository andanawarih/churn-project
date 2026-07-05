const form = document.getElementById("churn-form");
const resultPanel = document.getElementById("result-panel");
const needle = document.getElementById("needle");
const gaugeFill = document.getElementById("gauge-fill");
const riskLabel = document.getElementById("risk-label");
const riskValue = document.getElementById("risk-value");

// Smooth scroll untuk nav-links
document.querySelectorAll(".nav-link").forEach(link => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const targetId = link.getAttribute("href");
    const target = document.querySelector(targetId);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
});

// Form logic untuk fitur yang saling bergantung
const phoneService = form.querySelector('[name="PhoneService"]');
const multipleLines = form.querySelector('[name="MultipleLines"]');

const internetService = form.querySelector('[name="InternetService"]');
const internetFields = [
  "OnlineSecurity", "OnlineBackup", "DeviceProtection",
  "TechSupport", "StreamingTV", "StreamingMovies"
].map(name => form.querySelector(`[name="${name}"]`));

function updatePhoneLogic() {
  if (phoneService.value === "No") {
    multipleLines.value = "No phone service";
    multipleLines.disabled = true;
  } else {
    if (multipleLines.disabled) multipleLines.value = "No";
    multipleLines.disabled = false;
  }
}

function updateInternetLogic() {
  if (internetService.value === "No") {
    internetFields.forEach(f => {
      f.value = "No internet service";
      f.disabled = true;
    });
  } else {
    internetFields.forEach(f => {
      if (f.disabled) f.value = "No";
      f.disabled = false;
    });
  }
}

phoneService.addEventListener("change", updatePhoneLogic);
internetService.addEventListener("change", updateInternetLogic);

// Inisialisasi awal
updatePhoneLogic();
updateInternetLogic();

// Auto-calculate TotalCharges
const tenureInput = form.querySelector('[name="tenure"]');
const monthlyChargesInput = form.querySelector('[name="MonthlyCharges"]');
const totalChargesInput = form.querySelector('[name="TotalCharges"]');

// Jadikan TotalCharges readonly agar tidak bisa diedit manual
totalChargesInput.readOnly = true;
totalChargesInput.style.backgroundColor = "#e4e2d8"; // visual cue sedikit lebih gelap
totalChargesInput.style.cursor = "not-allowed";

function updateTotalCharges() {
  const tenure = parseFloat(tenureInput.value) || 0;
  const monthly = parseFloat(monthlyChargesInput.value) || 0;
  totalChargesInput.value = (tenure * monthly).toFixed(2);
}

tenureInput.addEventListener("input", updateTotalCharges);
monthlyChargesInput.addEventListener("input", updateTotalCharges);

// Set perhitungan awal
updateTotalCharges();

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Aktifkan elemen yang disabled sementara agar nilainya ikut terkirim ke backend
  const disabledElements = form.querySelectorAll(':disabled');
  disabledElements.forEach(el => el.disabled = false);

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  // Kembalikan ke state disabled
  disabledElements.forEach(el => el.disabled = true);

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await res.json();

  resultPanel.style.display = "block";
  resultPanel.scrollIntoView({ behavior: "smooth" });

  const prob = data.probability;
  riskValue.textContent = `${prob}%`;
  riskLabel.textContent = `Risiko ${data.risk_label}`;

  const arcLength = 315;
  const filled = (prob / 100) * arcLength;
  gaugeFill.setAttribute("stroke-dasharray", `${filled} ${arcLength}`);

  const color = prob >= 60 ? "#e2483c" : prob >= 30 ? "#e0b02a" : "#7bc94a";
  gaugeFill.setAttribute("stroke", color);

  const angle = (prob / 100) * 180 - 90;
  needle.style.transform = `rotate(${angle}deg)`;
});