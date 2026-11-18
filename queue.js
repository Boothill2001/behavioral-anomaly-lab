/* Inspect the saved XGB comparison and each reserved-quota decision offline. */
if (data.queue_research) {
  const report = data.queue_research, target = $("queue-research");
  const selected = report.selection.candidates.find(candidate => candidate.name === report.selection.main);
  const mainName = selected.method === "logistic" ? "Logistic regression" :
    "XGBoost · depth " + selected.depth + " · class weight " + selected.scale_pos_weight.toFixed(0);
  const jump = document.createElement("p"), link = document.createElement("a");
  jump.className = "jump";
  link.href = "#queue-research";
  link.textContent = t("XGBoost và quota review 45 + 5 ↓", "XGBoost and the 45 + 5 review queue ↓");
  jump.append(link);
  document.querySelector("header").append(jump);
  const policyName = key => key === "main_50" ? t("50 model chính + 0 IF", "50 main + 0 IF") : "45 main + 5 IF";
  const decisionName = key => ({
    review: t("Đưa vào review", "Review"), capacity_full: t("Quota nhánh đã đầy", "Branch cap reached"),
    below_cutoffs: t("Dưới ngưỡng", "Below cutoffs"), invalid_data: t("Dữ liệu lỗi", "Invalid data")
  })[key];
  const steps = t(["Input", "Điểm model", "Quota lúc đến", "Quyết định", "Nhãn hậu kiểm"],
                  ["Input", "Model scores", "Quota on arrival", "Decision", "Historical label"]);
  let step = 0;
  target.innerHTML = '<p class="kicker">' + t("NGHIÊN CỨU / DỮ LIỆU ĐÃ XEM", "RESEARCH / PREVIOUSLY INSPECTED DATA") + '</p>' +
    '<h2>' + t("Dành 5 lượt cho IF có giúp bắt thêm fraud?", "Do five reserved IF slots catch more fraud?") + '</h2>' +
    '<p class="muted">' + t("Model chính chọn trên dev", "Main model selected on dev") + ' · ' + mainName +
    '. ' + t("IF dùng 1.024 mẫu mỗi cây và tất cả feature. Quota hai nhánh được giữ riêng.",
             "IF uses 1,024 samples per tree and all features. Each branch keeps its reserved quota.") + '</p>' +
    '<p class="flag">' + t("Chính sách chọn trên dev", "Policy selected on dev") + ' · ' + policyName(report.selection.policy) + '</p>' +
    '<div class="controls jump"><label>Seed <select id="queue-seed"></select></label>' +
    '<label>' + t("Chính sách", "Policy") + ' <select id="queue-policy"></select></label>' +
    '<label>' + t("Loại ca", "Case group") + ' <select id="queue-group"></select></label>' +
    '<label>' + t("Giao dịch", "Transaction") + ' <select id="queue-case"></select></label></div>' +
    '<div id="queue-summary" aria-live="polite"></div>' +
    '<nav id="queue-flow" class="flow" aria-label="' + t("Flow review", "Review flow") + '"></nav>' +
    '<div id="queue-trace" aria-live="polite"></div>' +
    '<details><summary>' + t("Benchmark XGBoost trên dev và đánh giá", "XGBoost development and evaluation benchmark") + '</summary>' +
    '<div id="queue-models"></div></details>' +
    '<details><summary>' + t("Thời gian chấm batch CPU", "CPU batch scoring time") + '</summary>' +
    rows(t("Một batch 1.024 hàng, 10 lần đo sau warmup", "One 1,024-row batch, 10 measurements after warmup"),
      ["Model", "p50 ms", "p95 ms"],
      Object.entries(report.timings).map(([key, value]) => [key, value.p50_ms.toFixed(2), value.p95_ms.toFixed(2)])) + '</details>';
  const seedInput = $("queue-seed"), policyInput = $("queue-policy");
  const groupInput = $("queue-group"), caseInput = $("queue-case");
  options(seedInput, report.seed_values.map(seed => [seed, seed]), 42);
  options(policyInput, ["main_50", "reserved_45_5"].map(key => [key, policyName(key)]), report.selection.policy);
  const buttons = steps.map((title, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = (index + 1) + " · " + title;
    button.addEventListener("click", () => { step = index; draw(); });
    $("queue-flow").append(button);
    return button;
  });
  function draw() {
    const seed = seedInput.value, policy = policyInput.value;
    const keys = ["main_50_" + seed, "reserved_45_5_" + seed];
    const delta = report.paired[seed];
    $("queue-summary").innerHTML = rows(t("Đánh giá 12 giờ cuối", "Final 12-hour evaluation"),
      [t("Chính sách", "Policy"), t("Bắt được", "Caught"), t("Cảnh báo nhầm", "False alerts"),
        t("Đã duyệt", "Reviewed"), t("IF bắt được", "Caught via IF"), t("Overflow", "Overflow")],
      keys.map(key => {
        const value = report.results[key].arrival;
        return [policyName(key.startsWith("main_50") ? "main_50" : "reserved_45_5"),
          value.caught + " / 115", value.false_alerts, value.reviewed, value.branches.if.caught, value.overflow];
      })) + '<p class="muted">45 + 5 · ' + delta.rescued + ' ' + t("ca được cứu", "rescued") +
      ' · ' + delta.regressed + ' ' + t("ca phát sinh bỏ sót", "regressed") + '</p>';
    const candidateRows = report.selection.candidates.map(candidate => {
      const dev = candidate.runs.find(run => run.seed === Number(seed));
      const evaluation = report.models[candidate.name + "_" + seed];
      return [candidate.name, dev.caught + " / 96", evaluation.arrival.caught + " / 115",
        evaluation.arrival.false_alerts, evaluation.batch_top_k.budgets["50"].tp + " / 115",
        percent(evaluation.batch_top_k.ap)];
    });
    $("queue-models").innerHTML = rows(t("Cùng quota 50, cutoff chốt trên dev", "Same 50-slot quota, dev-frozen cutoffs"),
      ["Model", t("Dev replay", "Dev replay"), t("Đánh giá replay", "Evaluation replay"),
        t("Cảnh báo nhầm", "False alerts"), "Batch Top 50", "AP"], candidateRows);
    buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(step === index)));
    const record = report.cases.find(item => item.row_id === Number(caseInput.value));
    if (!record) {
      $("queue-trace").innerHTML = '<p class="flag">' + t("Không có ca trong nhóm này.", "No cases in this group.") + '</p>';
      return;
    }
    const key = policy + "_" + seed, trace = record.traces[key], settings = report.results[key];
    const view = $("queue-trace");
    let content = '';
    if (step === 0) {
      content = pair(t("Input · giao dịch", "Input · transaction"),
        {row_id: record.row_id, time_seconds: record.time_seconds, amount: record.amount},
        t("Output · kiểm tra dữ liệu", "Output · input validation"),
        {feature_count: Object.keys(record.features).length, all_finite: Object.values(record.features).every(Number.isFinite)});
      content += '<details><summary>' + t("Xem feature", "Inspect features") +
        '</summary><pre>' + dump(record.features) + '</pre></details>';
    } else if (step === 1) {
      content = pair(t("Điểm model", "Model scores"),
        {main: trace.main_score, isolation_forest: trace.if_score},
        t("Ngưỡng chốt trên dev", "Dev-frozen cutoffs"), {main: settings.cutoffs[0], isolation_forest: settings.cutoffs[1]});
    } else if (step === 2) {
      content = pair(t("Trước giao dịch", "Before arrival"),
        {main_slots: trace.main_before, if_slots: trace.if_before},
        t("Sau giao dịch", "After arrival"), {main_slots: trace.main_after, if_slots: trace.if_after});
      content += '<p class="muted">' + t("Giao dịch đạt ngưỡng model chính chỉ được xét ở nhánh chính, kể cả khi nhánh đã đầy.",
        "A transaction above the main cutoff stays in the main branch even when that branch is full.") + '</p>';
    } else if (step === 3) {
      content = pair(t("Input · nhánh và slot", "Input · branch and capacity"),
        {branch: trace.branch, main_slots: trace.main_before, if_slots: trace.if_before},
        t("Output · quyết định", "Output · decision"), {decision: trace.decision});
      content += '<p class="flag">' + decisionName(trace.decision) + '</p>';
    } else {
      content = pair(t("Quyết định replay", "Replay decision"),
        {reviewed: trace.decision === "review", branch: trace.branch},
        t("Nhãn lịch sử", "Historical label"), {fraud: record.label});
    }
    const comparisons = keys.map(method => {
      const other = record.traces[method];
      return [policyName(method.startsWith("main_50") ? "main_50" : "reserved_45_5"),
        other.branch, other.main_before + " / " + other.if_before, decisionName(other.decision)];
    });
    view.innerHTML = '<p class="kicker">ROW ' + record.row_id + ' / ' + steps[step] + '</p>' + content +
      rows(t("Cùng giao dịch, hai chính sách", "Same transaction under both policies"),
        [t("Chính sách", "Policy"), t("Nhánh", "Branch"), t("Slot chính / IF trước", "Main / IF slots before"), t("Quyết định", "Decision")],
        comparisons);
  }
  function chooseCase() {
    caseInput.replaceChildren();
    const ids = report.case_groups[groupInput.value];
    options(caseInput, ids.map(id => [id, "Row " + id]), ids[0]);
    draw();
  }
  function chooseSeed() {
    const seed = seedInput.value;
    groupInput.replaceChildren();
    const entries = [
      ["rescued_" + seed, t("Được cứu", "Rescued")],
      ["regressed_" + seed, t("Phát sinh bỏ sót", "Regressed")],
      ["if_false_alerts_" + seed, t("IF cảnh báo nhầm", "IF false alerts")],
      ["missed_both", t("Cả hai bỏ sót · seed 42", "Missed by both · seed 42")],
      ["caught_main", t("Model chính bắt được · seed 42", "Caught by main · seed 42")]
    ];
    options(groupInput, entries.map(([key, name]) => [key, name + " (" + report.case_groups[key].length + " " + t("ca mẫu", "examples") + ")"]),
      entries.find(([key]) => report.case_groups[key].length)?.[0] || entries[0][0]);
    chooseCase();
  }
  seedInput.addEventListener("change", chooseSeed);
  policyInput.addEventListener("change", draw);
  groupInput.addEventListener("change", chooseCase);
  caseInput.addEventListener("change", draw);
  chooseSeed();
  target.hidden = false;
}
