from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from medconsensus import __version__
from medconsensus.orchestrator import MedConsensusOrchestrator
from medconsensus.schemas import ConsensusReport, DISCLAIMER, SyntheticCaseRequest


app = FastAPI(
    title="MedConsensus",
    version=__version__,
    description="A2A-enabled multi-agent clinical reasoning demo for synthetic/de-identified patient cases.",
)
orchestrator = MedConsensusOrchestrator()


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MedConsensus</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6874;
      --line: #d8dde3;
      --panel: #ffffff;
      --page: #f6f7f9;
      --teal: #0f766e;
      --blue: #27548a;
      --amber: #b06114;
      --rose: #a33a3a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--page);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    .wrap {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }}
    .topbar {{
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
      letter-spacing: 0;
    }}
    .tagline {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .links {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .links a, button {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 12px;
      background: #ffffff;
      color: var(--ink);
      text-decoration: none;
      font: inherit;
      cursor: pointer;
    }}
    button.primary {{
      border-color: var(--teal);
      background: var(--teal);
      color: #ffffff;
      font-weight: 650;
    }}
    main {{
      padding: 22px 0 34px;
    }}
    .notice {{
      margin-bottom: 16px;
      border: 1px solid #e1c46f;
      border-left: 5px solid var(--amber);
      border-radius: 7px;
      padding: 12px 14px;
      background: #fff8e6;
      color: #4c3a18;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(320px, 0.95fr) minmax(360px, 1.25fr);
      gap: 16px;
      align-items: start;
    }}
    section {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .section-head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }}
    h2 {{
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .body {{
      padding: 16px;
    }}
    label {{
      display: block;
      margin: 0 0 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }}
    select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
    }}
    select {{
      min-height: 40px;
      padding: 8px 10px;
      margin-bottom: 12px;
    }}
    textarea {{
      min-height: 305px;
      resize: vertical;
      padding: 12px;
      line-height: 1.45;
    }}
    .actions {{
      margin-top: 12px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
    }}
    .agents {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }}
    .agent {{
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      min-height: 70px;
      background: #fbfcfd;
    }}
    .agent strong {{
      display: block;
      font-size: 13px;
    }}
    .agent span {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .output {{
      display: grid;
      gap: 12px;
    }}
    .result-block {{
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      background: #ffffff;
    }}
    .result-block h3 {{
      margin: 0 0 8px;
      font-size: 14px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
      color: #ffffff;
      background: var(--blue);
    }}
    .pill.high {{ background: var(--teal); }}
    .pill.moderate {{ background: var(--amber); }}
    .pill.low {{ background: var(--rose); }}
    ul {{
      margin: 8px 0 0;
      padding-left: 20px;
    }}
    li {{ margin: 5px 0; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
    }}
    .empty {{
      color: var(--muted);
      padding: 28px 16px;
      text-align: center;
    }}
    @media (max-width: 860px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .links {{ justify-content: flex-start; }}
      .grid {{ grid-template-columns: 1fr; }}
      .agents {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div>
        <h1>MedConsensus</h1>
        <p class="tagline">Multi-agent clinical reasoning that challenges diagnostic tunnel vision</p>
      </div>
      <nav class="links" aria-label="API links">
        <a href="/docs">Docs</a>
        <a href="/health">Health</a>
        <a href="/agent-card">Agent Card</a>
      </nav>
    </div>
  </header>
  <main class="wrap">
    <div class="notice">{DISCLAIMER} MedConsensus does not diagnose patients, recommend treatment, or replace clinician judgment.</div>
    <div class="notice">Instead of relying on a single AI response, MedConsensus makes specialist agents disagree, critique, and justify their reasoning before producing a final answer.</div>
    <div class="grid">
      <section>
        <div class="section-head">
          <h2>Synthetic Case Input</h2>
        </div>
        <div class="body">
          <label for="caseSelect">Demo case</label>
          <select id="caseSelect">
            <option value="heart">Heart failure vs pulmonary overlap</option>
            <option value="copd">COPD exacerbation with pneumonia</option>
            <option value="pe">Pulmonary embolism mimic</option>
          </select>
          <label for="caseText">Case text</label>
          <textarea id="caseText" spellcheck="false"></textarea>
          <div class="actions">
            <button class="primary" id="runButton" type="button">Run Consensus</button>
            <button id="resetButton" type="button">Reset Case</button>
            <span class="status" id="status">Ready</span>
          </div>
        </div>
      </section>
      <section>
        <div class="section-head">
          <h2>Disagreement to Consensus</h2>
          <span class="pill" id="confidencePill">waiting</span>
        </div>
        <div class="body">
          <div class="agents">
            <div class="agent"><strong>Orchestrator</strong><span>Routes case and debate</span></div>
            <div class="agent"><strong>Cardiology</strong><span>Heart and vascular lens</span></div>
            <div class="agent"><strong>Pulmonology</strong><span>Lung and airway lens</span></div>
            <div class="agent"><strong>Consensus</strong><span>Final synthesis</span></div>
          </div>
          <div id="output" class="output">
            <div class="empty">Run a synthetic case to see specialist assessments, debate, and consensus recommendations.</div>
          </div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const demoCases = {{
      heart: "Patient: 58F. Chief complaint: Progressively worsening dyspnea x 3 weeks, orthopnea, 2-pillow sleep. Vitals: HR 102, BP 148/92, RR 20, SpO2 94% on room air. Exam: Bilateral crackles at lung bases, JVD, 2+ pitting edema bilateral ankles, S3 gallop on auscultation. Labs: BNP 890 pg/mL, troponin I 0.04 ng/mL borderline, Cr 1.4 baseline 1.0, Na 133. CXR: Cardiomegaly, bilateral pleural effusions, Kerley B lines. History: HTN x 15y, T2DM, prior smoking 20 pack-years, no prior cardiac history documented.",
      copd: "Patient: 67M synthetic case. Chief complaint: Fever, productive cough, and worsening shortness of breath for 4 days. Vitals: HR 108, BP 132/78, RR 24, SpO2 90% on room air, temperature 38.4 C. Exam: Diffuse wheezes with focal right lower lobe crackles, no JVD, no pitting edema. Labs: WBC 15.8, BNP 90, troponin negative. CXR: Right lower lobe infiltrate with hyperinflation. History: COPD and former smoking.",
      pe: "Patient: 45F synthetic case. Chief complaint: Sudden dyspnea and pleuritic chest discomfort after a long flight. Vitals: HR 118, BP 124/76, RR 26, SpO2 91% on room air. Exam: Clear lungs, no S3, no JVD, mild unilateral calf tenderness. Labs: BNP 60, troponin 0.02, D-dimer elevated. CXR: No focal infiltrate, normal cardiac silhouette. History: Estrogen therapy, no known cardiac disease."
    }};

    const caseSelect = document.querySelector("#caseSelect");
    const caseText = document.querySelector("#caseText");
    const output = document.querySelector("#output");
    const statusEl = document.querySelector("#status");
    const confidencePill = document.querySelector("#confidencePill");

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }}[char]));
    }}

    function list(items) {{
      return `<ul>${{items.map((item) => `<li>${{escapeHtml(item)}}</li>`).join("")}}</ul>`;
    }}

    function confidenceClass(score) {{
      if (score >= 80) return "high";
      if (score >= 55) return "moderate";
      return "low";
    }}

    function loadCase() {{
      caseText.value = demoCases[caseSelect.value];
    }}

    function renderReport(report) {{
      const confidence = Number(report.consensus.confidence);
      const confidenceBand = confidenceClass(confidence);
      confidencePill.textContent = `${{confidence}}/100`;
      confidencePill.className = `pill ${{confidenceBand}}`;
      output.innerHTML = `
        <div class="result-block">
          <h3>Consensus</h3>
          <p><strong>Most likely:</strong> ${{escapeHtml(report.consensus.most_likely_diagnosis)}} <span class="pill ${{confidenceBand}}">${{confidence}}/100</span></p>
          <p><strong>Mode:</strong> ${{escapeHtml(report.metadata.mode)}}</p>
          <p><strong>Must-not-miss diagnoses:</strong> ${{escapeHtml(report.consensus.must_not_miss_diagnoses.join(", "))}}</p>
          <p><strong>ICD-10 codes</strong></p>
          ${{list(report.consensus.icd10_codes.map((item) => `${{item.code}} - ${{item.label}}`))}}
          <p><strong>Next questions</strong></p>
          ${{list(report.consensus.recommended_next_questions)}}
          <p><strong>Recommended tests</strong></p>
          ${{list(report.consensus.recommended_next_tests)}}
        </div>
        <div class="result-block">
          <h3>1. Three Specialist Opinions</h3>
          ${{report.specialist_assessments.map((item) => `
            <p><strong>${{escapeHtml(item.agent)}}:</strong> ${{escapeHtml(item.top_diagnosis)}}</p>
            ${{list(item.supporting_evidence)}}
          `).join("")}}
        </div>
        <div class="result-block">
          <h3>2. Critique Step</h3>
          ${{report.debate_summary.map((item) => `
            <p><strong>${{escapeHtml(item.from_agent)}}:</strong> ${{escapeHtml(item.challenge)}}</p>
            <p>${{escapeHtml(item.response_or_revision)}}</p>
          `).join("")}}
        </div>
        <div class="result-block">
          <h3>Raw JSON</h3>
          <pre>${{escapeHtml(JSON.stringify(report, null, 2))}}</pre>
        </div>
      `;
    }}

    async function runConsensus() {{
      statusEl.textContent = "Running debate...";
      confidencePill.textContent = "running";
      confidencePill.className = "pill";
      output.innerHTML = '<div class="empty">Specialists are assessing and debating the synthetic case.</div>';
      try {{
        const response = await fetch("/invoke", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            case_id: `ui-${{caseSelect.value}}`,
            synthetic: true,
            patient_case: caseText.value
          }})
        }});
        if (!response.ok) {{
          const error = await response.json();
          throw new Error(error.detail || "Request failed");
        }}
        renderReport(await response.json());
        statusEl.textContent = "Complete";
      }} catch (error) {{
        confidencePill.textContent = "error";
        confidencePill.className = "pill low";
        output.innerHTML = `<div class="result-block"><h3>Unable to run</h3><p>${{escapeHtml(error.message)}}</p></div>`;
        statusEl.textContent = "Error";
      }}
    }}

    caseSelect.addEventListener("change", loadCase);
    document.querySelector("#resetButton").addEventListener("click", loadCase);
    document.querySelector("#runButton").addEventListener("click", runConsensus);
    loadCase();
  </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "medconsensus", "version": __version__}


@app.get("/agent-card")
def agent_card() -> dict[str, object]:
    return {
        "name": "MedConsensus Orchestrator",
        "description": "Implements an A2A-style agent orchestration pattern with a discoverable agent-card endpoint compatible with Prompt Opinion integration requirements.",
        "version": __version__,
        "protocol": "a2a-compatible",
        "capabilities": [
            "synthetic_case_intake",
            "multi_specialist_independent_assessment",
            "structured_debate_round",
            "consensus_report",
            "llm_consensus_agent_when_configured",
            "llm_specialist_agents_when_configured",
            "llm_debate_round_when_configured",
            "phi_non_storage",
        ],
        "input_schema": SyntheticCaseRequest.model_json_schema(),
        "output_schema": ConsensusReport.model_json_schema(),
        "endpoints": {"health": "/health", "invoke": "/invoke", "tasks": "/tasks"},
        "safety": {
            "synthetic_only": True,
            "stores_phi": False,
            "disclaimer": DISCLAIMER,
            "recommendation_scope": "Next clinical questions/tests only; no treatment prescribing.",
        },
        "llm_configuration": {
            "provider_env": "MEDCONSENSUS_LLM_PROVIDER=anthropic",
            "enabled_env": "MEDCONSENSUS_USE_LLM=true",
            "api_key_env": "ANTHROPIC_API_KEY",
            "fallback_mode": "deterministic_fallback",
            "llm_mode": "llm_multi_agent",
        },
        "agents": [
            "MedConsensus Orchestrator",
            "General Medicine Agent",
            "Cardiology Agent",
            "Pulmonology Agent",
            "Consensus Agent",
        ],
    }


@app.post("/invoke", response_model=ConsensusReport)
def invoke(request: SyntheticCaseRequest) -> ConsensusReport:
    try:
        return orchestrator.invoke(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tasks", response_model=ConsensusReport)
def tasks(request: SyntheticCaseRequest) -> ConsensusReport:
    return invoke(request)
