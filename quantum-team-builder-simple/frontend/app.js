"use strict";

// We display saved evidence only. This page never starts a solver or a backend job.
const DATA_URL = "./data/results.json";
const SVG_NS = "http://www.w3.org/2000/svg";
const COLORS = { navy: "#143242", teal: "#007f7a", muted: "#5b6c73", line: "#dce5e7", soft: "#f5f8f8" };
const METHOD_NAMES = { exact: "Exact enumeration", random: "Random feasible search", qaoa: "QAOA" };
const SKILL_NAMES = { Programming: "Programming", Python: "Python", Design: "Design", Communication: "Communication" };
const state = { data: null, selectedCase: null, method: "qaoa", chartSerial: 0 };
const $ = (id) => document.getElementById(id);
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const skillName = (key) => SKILL_NAMES[key] || String(key);
const methodName = (key) => METHOD_NAMES[key] || String(key);

function number(value, digits = 4) {
  if (!finite(value)) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value);
}
function seconds(value) {
  if (!finite(value)) return "—";
  return `${value > 0 && value < 0.001 ? value.toExponential(2) : number(value, 4)} s`;
}
function percent(value) { return finite(value) ? `${number(value * 100, 2)}%` : "—"; }
function precise(value) { return finite(value) ? String(value) : "—"; }
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}
function svgElement(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== undefined) node.textContent = String(text);
  return node;
}
function svgChart(width, height, title, description) {
  const id = `chart-${++state.chartSerial}`;
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-labelledby": `${id}-title ${id}-desc` });
  svg.append(svgElement("title", { id: `${id}-title` }, title), svgElement("desc", { id: `${id}-desc` }, description));
  return svg;
}
function textSvg(svg, x, y, text, attrs = {}) {
  svg.append(svgElement("text", { x, y, fill: COLORS.muted, "font-size": 12, ...attrs }, text));
}
function emptyChart(container, message) {
  container.replaceChildren(element("p", "chart-empty", message));
}
function table(headers, rows, { caption = "", numeric = [] } = {}) {
  const result = element("table");
  if (caption) result.append(element("caption", "sr-only", caption));
  const thead = element("thead");
  const heading = element("tr");
  headers.forEach((name, index) => {
    const th = element("th", numeric.includes(index) ? "number-cell" : "", name);
    th.scope = "col";
    heading.append(th);
  });
  thead.append(heading);
  const tbody = element("tbody");
  rows.forEach((row) => {
    const tr = element("tr", row.className || "");
    const cells = Array.isArray(row) ? row : row.cells;
    cells.forEach((value, index) => {
      const td = element("td", numeric.includes(index) ? "number-cell" : "");
      if (value instanceof Node) td.append(value);
      else td.textContent = String(value ?? "—");
      tr.append(td);
    });
    tbody.append(tr);
  });
  result.append(thead, tbody);
  return result;
}
function fact(container, title, value) {
  const div = element("div");
  div.append(element("dt", "", title), element("dd", "", value));
  container.append(div);
}
function resultStatus(result) {
  if (!result) return { text: "No saved result", type: "warning" };
  const statuses = {
    optimal: ["Certified optimum", "success"], completed: ["Completed", "success"], success: ["Completed", "success"],
    timeout: ["Time limit reached", "warning"], error: ["Execution error", "error"],
    infeasible: ["Infeasible return", "warning"], invalid_solution: ["Infeasible return", "warning"],
    no_feasible_samples: ["No feasible final sample", "warning"], no_feasible_sample: ["No feasible final sample", "warning"]
  };
  let [text, type] = statuses[result.status] || [`Recorded status: ${String(result.status ?? "Not provided")}`, "warning"];
  if (result.returned?.feasible === false && type === "success") { text = "Infeasible return"; type = "warning"; }
  return { text, type };
}
function badge(result) {
  const status = resultStatus(result);
  return element("span", `status-badge ${status.type}`, status.text);
}
function activeCase() { return state.data.cases.find((item) => item.id === state.selectedCase); }
function quantumResult(item) { return item.results.find((result) => result.solver === "qaoa"); }
function countEntries(result) {
  return Object.entries(result?.qaoa?.counts || {}).filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

function validateData(data) {
  const fail = (message) => { throw new Error(`Invalid data structure: ${message}`); };
  if (!data || data.schema_version !== 1) fail("We require schema_version=1.");
  if (!data.pool || !Number.isInteger(data.pool.rows) || !Array.isArray(data.pool.skills) || !Array.isArray(data.pool.head)) fail("We could not find the source pool or CSV preview.");
  if (data.pool.synthetic !== true) fail("We require explicitly labeled synthetic data for this display.");
  if (!Array.isArray(data.cases) || data.cases.length === 0) fail("We could not find a case to display.");
  const caseIds = new Set();
  for (const item of data.cases) {
    if (typeof item.id !== "string" || caseIds.has(item.id)) fail("Case IDs are missing or duplicated.");
    caseIds.add(item.id);
    if (!Number.isInteger(item.n) || item.n <= 0 || item.n % 2 !== 0 || item.team_size !== item.n / 2) fail(`${item.id}: participant count and team size disagree.`);
    if (!Array.isArray(item.ids) || item.ids.length !== item.n || !Array.isArray(item.skills) || !item.skills.length) fail(`${item.id}: participant or skill columns are incomplete.`);
    if (!Array.isArray(item.scores) || item.scores.length !== item.n || item.scores.some((row) => !Array.isArray(row) || row.length !== item.skills.length || row.some((v) => !finite(v)))) fail(`${item.id}: the skill matrix is invalid.`);
    const q = item.qubo;
    if (!q || !finite(q.constant) || !Array.isArray(q.linear) || q.linear.length !== item.n || q.linear.some((v) => !finite(v)) || !Array.isArray(q.quadratic) || q.quadratic.length !== item.n || q.quadratic.some((row) => !Array.isArray(row) || row.length !== item.n || row.some((v) => !finite(v)))) fail(`${item.id}: QUBO coefficients are incomplete.`);
    if (!item.ising || !finite(item.ising.offset) || !Array.isArray(item.ising.terms) || item.ising.terms.some((term) => typeof term.label !== "string" || !finite(term.coefficient))) fail(`${item.id}: Ising records are incomplete.`);
    if (!Array.isArray(item.results)) fail(`${item.id}: solver results are missing.`);
    for (const result of item.results) {
      if (result.solver === "qaoa") {
        for (const [bits, count] of Object.entries(result.qaoa?.counts || {})) {
          if (!/^[01]+$/.test(bits) || bits.length !== item.n || !Number.isInteger(count) || count < 0) fail(`${item.id}: measurement counts are invalid.`);
        }
      }
    }
  }
}

function renderCaseControls() {
  const controls = $("case-controls");
  controls.replaceChildren();
  state.data.cases.forEach((item) => {
    const button = element("button", "case-button", `${number(item.n, 0)} people`);
    button.type = "button";
    button.setAttribute("aria-pressed", String(item.id === state.selectedCase));
    button.addEventListener("click", () => {
      if (state.selectedCase === item.id) return;
      state.selectedCase = item.id;
      renderCase();
      [...controls.children].forEach((child, i) => child.setAttribute("aria-pressed", String(state.data.cases[i].id === state.selectedCase)));
    });
    controls.append(button);
  });
}

function renderBackground(item) {
  const pool = state.data.pool;
  const sourceHash = pool.sha256 || pool.source_hash;
  $("pool-count").textContent = number(pool.rows, 0);
  $("case-count").textContent = number(item.n, 0);
  $("team-size-label").textContent = `Two teams of ${number(item.team_size, 0)} people`;
  $("case-description").textContent = `We select this fixed ${number(item.n, 0)}-person case from ${number(pool.rows, 0)} synthetic records. Repeated measurements do not add participants.`;
  $("case-tag").textContent = `${number(item.n, 0)} people · ${number(item.skills.length, 0)} skills`;
  $("pool-table").replaceChildren(table(["ID", ...pool.skills.map(skillName)], pool.head.map((row) => [String(row.id), ...pool.skills.map((skill) => number(row[skill]))]), { caption: "First rows of the synthetic source CSV", numeric: pool.skills.map((_, i) => i + 1) }));
  $("case-table").replaceChildren(table(["ID", ...item.skills.map(skillName)], item.ids.map((id, index) => [String(id), ...item.scores[index].map((score) => number(score))]), { caption: "All participants and skill scores in the selected case", numeric: item.skills.map((_, i) => i + 1) }));
  $("source-note").textContent = sourceHash ? `We preserve the source fingerprint (SHA-256): ${String(sourceHash).slice(0, 16)}…` : "We display the saved source-pool preview.";
  $("source-note").title = sourceHash ? String(sourceHash) : "";
  const measured = countEntries(quantumResult(item)).reduce((sum, [, count]) => sum + count, 0);
  $("shot-count").textContent = measured > 0 ? number(measured, 0) : "No samples";
}

function renderMethodSelector(item) {
  const select = $("skill-method");
  select.replaceChildren();
  if (!item.results.some((result) => result.solver === state.method)) state.method = item.results[0]?.solver || "";
  item.results.forEach((result) => {
    const option = element("option", "", methodName(result.solver));
    option.value = result.solver;
    select.append(option);
  });
  select.value = state.method;
  select.disabled = item.results.length === 0;
  renderSkills(item);
}

function renderSkills(item) {
  const result = item.results.find((record) => record.solver === state.method);
  const returned = result?.returned;
  $("team-result-status").replaceChildren(badge(result));
  $("team-members").replaceChildren();
  if (!returned) {
    emptyChart($("skill-chart"), "We have no saved assignment for this run. Select another method to inspect its returned teams.");
    return;
  }
  for (const [key, label, className] of [["team_1", "Team 1", "team-one"], ["team_2", "Team 2", "team-two"]]) {
    const row = element("div", "team-line");
    row.append(element("strong", className, label), element("span", "", Array.isArray(returned[key]) ? returned[key].join(", ") || "(empty team)" : "No saved participant list"));
    $("team-members").append(row);
  }
  const values = item.skills.map((skill) => [returned.skill_sums?.team_1?.[skill], returned.skill_sums?.team_2?.[skill]]);
  if (values.some((row) => row.some((v) => !finite(v)))) { emptyChart($("skill-chart"), "We have incomplete skill totals for this assignment."); return; }
  const width = 560, height = 52 + item.skills.length * 80, left = 135, right = 46, plotWidth = width - left - right;
  const max = Math.max(1, ...values.flat());
  const description = item.skills.map((skill, i) => `${skillName(skill)}: Team 1 ${number(values[i][0])}, Team 2 ${number(values[i][1])}`).join("; ");
  const svg = svgChart(width, height, `Skill totals in the assignment returned by ${methodName(result.solver)}`, description);
  for (let tick = 0; tick <= 4; tick++) {
    const x = left + plotWidth * tick / 4;
    svg.append(svgElement("line", { x1: x, x2: x, y1: 22, y2: height - 34, stroke: COLORS.line }));
    textSvg(svg, x, height - 12, number(max * tick / 4, 1), { "text-anchor": "middle", "font-size": 11 });
  }
  item.skills.forEach((skill, i) => {
    const y = 24 + i * 80;
    textSvg(svg, left - 13, y + 29, skillName(skill), { "text-anchor": "end", fill: COLORS.navy, "font-size": 13 });
    values[i].forEach((value, team) => {
      const bar = svgElement("rect", { x: left, y: y + team * 25, width: Math.max(0, value) / max * plotWidth, height: 17, rx: 3, fill: team ? COLORS.teal : COLORS.navy });
      bar.append(svgElement("title", {}, `${skillName(skill)}, Team ${team + 1}: ${precise(value)}`));
      svg.append(bar);
      textSvg(svg, left + Math.max(0, value) / max * plotWidth + 7, y + team * 25 + 13, number(value), { fill: COLORS.navy, "font-size": 12 });
    });
  });
  $("skill-chart").replaceChildren(svg);
}

function coefficientColor(value, max) {
  if (max === 0) return "#f7faf9";
  const start = [247, 250, 249], end = value < 0 ? [37, 75, 101] : [0, 133, 123];
  const amount = Math.min(1, Math.abs(value) / max);
  return `rgb(${start.map((channel, i) => Math.round(channel + amount * (end[i] - channel))).join(",")})`;
}
function renderQubo(item) {
  const q = item.qubo, n = item.n;
  $("penalty-description").textContent = `We use P=${number(item.penalty)}. A deviation from ${number(item.team_size, 0)} people per team adds a squared size penalty.`;
  $("qubo-constant").textContent = number(q.constant);
  $("qubo-constant").title = precise(q.constant);
  $("qubo-variables").textContent = number(n, 0);
  const side = 414, left = 49, top = 40, cell = side / n, max = Math.max(...q.linear.map(Math.abs), ...q.quadratic.flat().map(Math.abs));
  const svg = svgChart(490, 485, `Upper-triangle QUBO coefficient matrix for the ${n}-person case`, "We show linear coefficients on the diagonal and quadratic coefficients above it. We leave the lower triangle blank. The adjacent table contains every coefficient.");
  const coefficientRows = [];
  for (let i = 0; i < n; i++) {
    textSvg(svg, left + (i + 0.5) * cell, top - 13, i, { "text-anchor": "middle", "font-size": n > 10 ? 10 : 12 });
    textSvg(svg, left - 14, top + (i + 0.5) * cell + 4, i, { "text-anchor": "end", "font-size": n > 10 ? 10 : 12 });
    coefficientRows.push([`h${i}`, String(item.ids[i]), number(q.linear[i], 6)]);
    for (let j = 0; j < n; j++) {
      const value = i === j ? q.linear[i] : q.quadratic[i][j];
      const rect = svgElement("rect", { x: left + j * cell, y: top + i * cell, width: cell - 1, height: cell - 1, fill: j < i ? COLORS.soft : coefficientColor(value, max) });
      if (j >= i) rect.append(svgElement("title", {}, `${i === j ? `h${i}` : `J${i},${j}`} (${item.ids[i]}${i === j ? "" : `, ${item.ids[j]}`}): ${precise(value)}`));
      svg.append(rect);
      if (j > i) coefficientRows.push([`J${i},${j}`, `${item.ids[i]} · ${item.ids[j]}`, number(value, 6)]);
      if (j >= i && n <= 8) textSvg(svg, left + (j + 0.5) * cell, top + (i + 0.5) * cell + 4, number(value, 1), { "text-anchor": "middle", "font-size": 12, fill: max > 0 && Math.abs(value) / max > 0.62 ? "#fff" : COLORS.navy });
    }
  }
  textSvg(svg, left + side / 2, top + side + 24, "Variable indices i, j (input order, starting at 0)", { "text-anchor": "middle", "font-size": 11 });
  $("qubo-chart").replaceChildren(svg);
  $("qubo-caption").textContent = `We show ${n} linear coefficients and ${number(n * (n - 1) / 2, 0)} upper-triangle positions, including zero coefficients.`;
  $("coefficient-table").replaceChildren(table(["Coefficient", "Participants", "Value"], coefficientRows, { caption: "All linear and upper-triangle quadratic QUBO coefficients", numeric: [2] }));
}

function renderIsing(item) {
  $("ising-offset").textContent = number(item.ising.offset);
  $("ising-offset").title = precise(item.ising.offset);
  $("ising-term-count").textContent = number(item.ising.terms.length, 0);
  const rows = item.ising.terms.map((term) => [element("span", "mono", term.label), number(term.coefficient, 6)]);
  $("ising-terms").replaceChildren(table(["Pauli operator", "Coefficient"], rows, { caption: "Saved Ising terms and coefficients, with the constant offset shown separately", numeric: [1] }));
}

function renderCounts(item, result) {
  const entries = countEntries(result), total = entries.reduce((sum, [, count]) => sum + count, 0);
  const container = $("count-chart");
  if (total === 0) {
    emptyChart(container, "We have no saved measurement counts for this run, so we do not draw a distribution.");
    $("histogram-count").textContent = "No saved measurement samples";
    $("count-caption").textContent = "We preserve timeouts, errors, and absent samples without substituting successful measurements.";
    return;
  }
  const top = entries.slice(0, 10), width = 880, left = item.n > 12 ? 190 : 122, right = 150, topMargin = 20, rowHeight = 36;
  const height = topMargin + top.length * rowHeight + 38, plotWidth = width - left - right, max = top[0][1];
  const svg = svgChart(width, height, "Ten most frequent quantum measurement outcomes", `We saved ${total} measurements. Bars show counts; every percentage uses all ${total} measurements as its denominator. ${top.map(([bits, count]) => `${bits}: ${count} shots, ${percent(count / total)}`).join("; ")}.`);
  for (let tick = 0; tick <= 4; tick++) {
    const x = left + plotWidth * tick / 4;
    svg.append(svgElement("line", { x1: x, x2: x, y1: topMargin - 3, y2: height - 35, stroke: COLORS.line }));
    textSvg(svg, x, height - 14, number(max * tick / 4, 1), { "text-anchor": "middle", "font-size": 12 });
  }
  top.forEach(([bits, count], index) => {
    const y = topMargin + index * rowHeight;
    textSvg(svg, left - 14, y + 18, bits, { "text-anchor": "end", fill: COLORS.navy, "font-size": 13, "font-family": "ui-monospace,SFMono-Regular,Consolas,monospace" });
    const bar = svgElement("rect", { x: left, y, width: count / max * plotWidth, height: 25, rx: 3, fill: COLORS.teal });
    bar.append(svgElement("title", {}, `${bits}: ${count} shots, ${percent(count / total)} of all measurements`));
    svg.append(bar);
    textSvg(svg, left + plotWidth + 15, y + 18, `${number(count, 0)} shots · ${percent(count / total)}`, { fill: COLORS.navy, "font-size": 12 });
  });
  container.replaceChildren(svg);
  const shown = top.reduce((sum, [, count]) => sum + count, 0), remainder = total - shown;
  $("histogram-count").textContent = `Denominator: all ${number(total, 0)} shots`;
  $("count-caption").textContent = `We show ${top.length} outcomes. The remaining ${number(entries.length - top.length, 0)} outcomes account for ${number(remainder, 0)} shots (${percent(remainder / total)}). The horizontal axis shows counts; all percentages use the full measurement total.`;
}

function renderQuantum(item) {
  const result = quantumResult(item), status = resultStatus(result), q = result?.qaoa || {}, b = result?.budgets || {}, resources = q.resources || {};
  const statusBox = $("qaoa-status");
  statusBox.className = `run-notice ${status.type}`;
  let message = `Our saved QAOA status: ${status.text}. `;
  if (result?.status === "timeout") message += `We reached ${finite(b.wall_seconds_limit) ? `the ${number(b.wall_seconds_limit)}-second` : "the configured"} time limit. We preserve this stopping status without marking the run completed.`;
  else if (result?.status === "error") message += `We preserve the execution error: ${String(result.error || q.error || "No detailed error was saved")}.`;
  else if (result?.returned?.feasible === false) message += "We received unequal team sizes. We do not treat this energy as a better valid partition.";
  else if (result?.returned?.feasible === true) message += "We received an assignment with equal team sizes. This delivery outcome differs from the success probability of a single measurement.";
  statusBox.textContent = message;
  const facts = $("qaoa-facts"); facts.replaceChildren();
  fact(facts, "QAOA depth p", number(q.p, 0));
  fact(facts, "Independent-sample feasible frequency", percent(q.feasible_sample_mass));
  fact(facts, "Independent-sample optimal frequency", percent(q.optimal_sample_mass));
  fact(facts, "Measured runtime", seconds(result?.runtime_seconds));
  fact(facts, "Qubits", number(resources.n_qubits, 0));
  fact(facts, "Circuit depth", number(resources.depth, 0));
  fact(facts, "CX gates", number(resources.cx_count, 0));
  fact(facts, "Objective evaluations used", number(b.objective_evaluations, 0));
  const detail = $("qaoa-details"); detail.replaceChildren();
  const stopNames = { wall_time_limit: "Wall-time limit reached", evaluation_limit: "Objective-evaluation limit reached", optimizer_terminated: "Optimizer terminated (no optimality proof)", solver_error: "Solver error" };
  const parameterText = q.parameters === null || q.parameters === undefined ? "No saved optimized parameters" : JSON.stringify(q.parameters);
  for (const [label, value] of [
    ["Random seed", number(q.seed, 0)], ["Independent-measurement seed", number(q.diagnostic_seed, 0)],
    ["Parameter names and order", Array.isArray(q.parameter_names) ? q.parameter_names.join(", ") : "Not provided"], ["Saved parameters", parameterText],
    ["Shots per sampling call", number(b.shots, 0)], ["Objective-evaluation limit", number(b.objective_evaluations_limit, 0)],
    ["Wall-time limit", seconds(b.wall_seconds_limit)], ["Stopping reason", stopNames[q.stop_reason] || String(q.stop_reason || "Not provided")],
    ["Circuit-resource convention", "We count gates and depth using the declared basis gates. These are not measured hardware timings."]
  ]) fact(detail, label, value);
  $("sampling-note").textContent = "We use a separate finite-shot sample for this chart and both frequencies. The returned teams come from Qiskit’s own final sample; we do not replace them with the independent sample or exact answer. One run per size does not establish general performance or quantum speedup.";
  renderCounts(item, result);
}

function renderComparison(item) {
  const rows = item.results.map((result) => {
    const r = result.returned;
    const feasible = r?.feasible === true ? "Yes" : r?.feasible === false ? "No" : "No return";
    const gap = r?.feasible === false ? "N/A" : finite(result.gap) ? number(result.gap, 6) : "—";
    const budget = result.solver === "qaoa" ? `${number(result.budgets?.objective_evaluations, 0)} objective evaluations` : `${number(result.budgets?.candidate_evaluations, 0)} candidates`;
    return { className: r?.feasible === false ? "row-invalid" : "", cells: [methodName(result.solver), badge(result), number(r?.E), number(r?.C), number(r?.B), feasible, gap, seconds(result.runtime_seconds), budget] };
  });
  $("comparison-table").replaceChildren(table(["Method", "Status", "Energy E ↓", "Within-team C", "Imbalance B ↓", "Equal-size feasible", "Absolute energy gap ΔE ↓", "Runtime", "Budget used"], rows, { caption: "Actual classical and quantum returns for the selected case", numeric: [2, 3, 4, 6, 7] }));
  const verification = item.verification, card = $("verification-card");
  const exact = item.results.find((result) => result.solver === "exact");
  card.replaceChildren(); card.className = "verification-card";
  if (!verification) {
    if (item.n === 20 && exact?.status === "optimal" && exact.returned?.feasible === true) {
      card.classList.add("passed");
      card.append(element("h3", "", "We verify the 20-person feasible optimum"));
      card.append(element("p", "", `We enumerate all ${number(exact.budgets?.candidate_evaluations, 0)} equal-size labeled assignments and obtain the feasible optimum E* = ${number(exact.returned.E)}. We use this exact reference to calculate absolute energy gaps.`));
    } else {
      card.append(element("h3", "", "We use separate checks at each scale"), element("p", "", "We show the saved classical reference and actual returns. We do not infer checks that were not performed."));
    }
    const small = state.data.cases.find((record) => record.n === 6 && record.verification?.passed === true);
    if (small) card.append(element("p", "", `We perform full encoding equivalence checks on the 6-person case (${number(small.verification.states_checked, 0)} states). We do not repeat those checks across all 2^${item.n} binary states in this case.`));
    if (Number.isInteger(exact?.optimal_assignment_count)) card.append(element("p", "small-note", `We retain ${number(exact.optimal_assignment_count, 0)} optimal labeled assignments. Swapping both team names gives the same unlabeled partition.`));
    return;
  }
  card.classList.add(verification.passed === true ? "passed" : "failed");
  card.append(element("h3", "", verification.passed === true ? "Our full encoding equivalence check passed" : "Our full encoding equivalence check failed"));
  card.append(element("p", "", "We compare the direct objective, manual QUBO, Qiskit quadratic objective, and Ising energy including its offset. Encoding agreement does not guarantee that QAOA always returns an optimum."));
  const values = element("div", "verification-values");
  for (const [label, value] of [["States checked", number(verification.states_checked, 0)], ["Maximum absolute error", precise(verification.max_error)], ["Tolerance", precise(verification.tolerance)]]) {
    const span = element("span", "", `${label}: `); span.append(element("strong", "", value)); values.append(span);
  }
  card.append(values);
  if (Number.isInteger(exact?.optimal_assignment_count)) card.append(element("p", "small-note", `We retain ${number(exact.optimal_assignment_count, 0)} optimal labeled assignments. Swapping both team names gives the same unlabeled partition.`));
}

function renderCase() {
  const item = activeCase();
  renderBackground(item);
  renderMethodSelector(item);
  renderQubo(item);
  renderIsing(item);
  renderQuantum(item);
  renderComparison(item);
}

$("skill-method").addEventListener("change", (event) => {
  state.method = event.target.value;
  renderSkills(activeCase());
});

(async function load() {
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`We could not load data/results.json (HTTP ${response.status}).`);
    const data = await response.json();
    validateData(data);
    state.data = data;
    state.selectedCase = data.cases.some((item) => item.id === "n6") ? "n6" : data.cases[0].id;
    renderCaseControls();
    renderCase();
    $("load-status").hidden = true;
    $("experiment").hidden = false;
  } catch (error) {
    const box = $("load-status");
    box.className = "load-status error";
    box.setAttribute("role", "alert");
    box.replaceChildren(element("strong", "", "We cannot display the experiment results yet."), element("p", "", String(error.message || error)), element("p", "", "We need the generated data/results.json file and a local HTTP server to open this page. We do not substitute example values for missing experiments."));
    $("experiment").hidden = true;
  }
})();
