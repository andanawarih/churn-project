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

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

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