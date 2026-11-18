/* Embedded in index.html so saved research can be inspected offline. */
if (data.if_research) {
  const research = data.if_research, target = $("if-research");
  function label(method) {
    if (method === "logistic") return "Logistic regression";
    const seed = method.split("_").at(-1);
    if (method.startsWith("original_if")) return t("IF gốc", "Original IF") + " · " + seed;
    if (method.startsWith("selected_if")) return t("IF chọn trên dev", "Dev-selected IF") + " · " + seed;
    return "Logistic + IF · " + seed;
  }
  const groupNames = {
    original_false_alerts: t("20 cảnh báo nhầm của IF gốc", "20 original IF false alerts"),
    original_missed_frauds: t("20 fraud bị IF gốc bỏ sót", "20 original IF missed frauds"),
    still_missed: t("Vẫn bị bỏ sót", "Still missed")
  };
  const jump = document.createElement("p"), anchor = document.createElement("a");
  jump.className = "jump";
  anchor.href = "#if-research";
  anchor.textContent = t("Vì sao Isolation Forest bỏ sót fraud? ↓", "Why did Isolation Forest miss fraud? ↓");
  jump.append(anchor);
  document.querySelector("header").append(jump);
  const recommendation = research.recommendation === "keep_logistic"
    ? t("Giữ Logistic Regression làm mặc định", "Keep logistic regression as the default")
    : t("Bản kết hợp đạt tiêu chí nghiên cứu", "The combined candidate passed the research gate");
  const selectionRows = research.selection.grid.map(config => [
    config.config.max_samples, config.config.max_features,
    config.runs.map(run => run.metrics.budgets["50"].tp).join(" / "),
    config.mean_dev_caught.toFixed(2), config.std_dev_caught.toFixed(2)]);
  const windowRows = [
    ...research.dev_diagnostics.original_if_42.map(window => ({period:"Dev", ...window})),
    ...research.diagnostics.original_if_42.map(window => ({period:t("Đánh giá", "Evaluation"), ...window}))
  ].flatMap(window => ["0", "1"].map(key => {
    const group = window.score_by_label[key];
    return [window.period, window.hours.join("–"), key === "1" ? "Fraud" : t("Hợp lệ", "Legitimate"),
      group.n, group.p50 === null ? "—" : group.p50.toFixed(4),
      group.p95 === null ? "—" : group.p95.toFixed(4),
      key === "1" ? [50, 100, 200].map(k => window.fraud_top_k[k]).join(" / ") : "—"];
  }));
  target.innerHTML =
    '<p class="kicker">' + t("NGHIÊN CỨU / DỮ LIỆU ĐÃ XEM", "RESEARCH / PREVIOUSLY INSPECTED DATA") + '</p>' +
    '<h2>' + t("Vì sao Isolation Forest bỏ sót fraud?", "Why did Isolation Forest miss fraud?") + '</h2>' +
    '<p class="muted">' + t(
      "6 cấu hình × 3 seed, chọn trên dev trước khi chấm lại 12 giờ cuối. Bản kết hợp học thêm điểm IF tạo bằng cross-fitting 5 fold trong train.",
      "Six configurations across three seeds, selected on dev before reevaluating the final 12 hours. The combined classifier adds an IF score cross-fitted across five training folds.") + '</p>' +
    '<p class="flag">' + recommendation + '</p>' +
    '<div class="controls jump"><label>Seed <select id="research-seed"></select></label>' +
    '<label>' + t("Loại ca", "Case group") + ' <select id="research-group"></select></label>' +
    '<label>' + t("Giao dịch", "Transaction") + ' <select id="research-case"></select></label></div>' +
    '<div id="research-summary" aria-live="polite"></div><div id="research-case-view" aria-live="polite"></div>' +
    '<details><summary>' + t("Các cấu hình đã thử trên dev", "Every configuration tried on dev") + '</summary>' +
    rows(t("Top 50 mỗi batch", "Top 50 per batch"),
      ["max_samples", "max_features", t("Bắt được · 42/43/44", "Caught · 42/43/44"), t("Trung bình", "Mean"), t("Độ lệch seed", "Seed std.")],
      selectionRows) +
    '<p class="muted">' + t("Cấu hình chọn", "Selected configuration") + ' · ' +
      escapeText(JSON.stringify(research.selection.chosen_config)) + '</p></details>' +
    '<details><summary>' + t("Theo cửa sổ và nhãn · điểm IF gốc", "Original IF scores by window and label") + '</summary>' +
    rows(t("Điểm cao được xếp trước", "Higher scores rank first"),
      [t("Giai đoạn", "Period"), t("Giờ", "Hours"), t("Nhãn", "Label"), "n", "p50", "p95", "Fraud Top 50/100/200"], windowRows) + '</details>';
  const seedInput = $("research-seed"), groupInput = $("research-group"), caseInput = $("research-case");
  options(seedInput, research.selection.seeds.map(seed => [seed, seed]), 42);
  function groupLabel(key) {
    if (groupNames[key]) return groupNames[key];
    const prefix = key.startsWith("combined") ? "Logistic + IF" : t("IF chọn trên dev", "Dev-selected IF");
    return prefix + " · " + key.match(/_(42|43|44)_/)[1] + " · " +
      (key.endsWith("rescued") ? t("được cứu", "rescued") : t("phát sinh bỏ sót", "regressed"));
  }
  const groups = Object.entries(research.case_groups).filter(([, ids]) => ids.length);
  options(groupInput, groups.map(([key]) => [key, groupLabel(key)]), "original_missed_frauds");
  const decisions = {
    review: t("Duyệt", "Review"), below_cutoff: t("Dưới ngưỡng", "Below cutoff"),
    capacity_full: t("Hết quota", "Queue cap reached")
  };
  function renderResearch() {
    const seed = seedInput.value;
    const methods = ["original_if_" + seed, "selected_if_" + seed, "logistic", "combined_" + seed];
    const summaryRows = methods.map(method => {
      const result = research.results[method];
      return [label(method), result.dev.budgets["50"].tp + " / 96",
        result.metrics.budgets["50"].tp + " / 115", result.arrival.caught + " / 115",
        result.arrival.reviewed, result.arrival.false_alerts];
    });
    const paired = research.paired_arrival_changes["combined_" + seed + "_vs_logistic"];
    $("research-summary").innerHTML = rows(
      t("Cùng tối đa 100 lượt duyệt trên 12 giờ đánh giá", "Up to 100 reviews over the same final 12 hours"),
      [t("Phương pháp", "Method"), "Dev Top 50", t("Fraud cuối batch", "Batch catches"),
        t("Fraud lúc đến", "Arrival catches"), t("Đã duyệt", "Reviewed"), t("Cảnh báo nhầm", "False alerts")], summaryRows) +
      '<p class="muted">' + t("Bản kết hợp so với Logistic khi replay", "Combined vs. logistic on arrival") +
      ' · ' + paired.rescued + ' ' + t("ca được cứu", "rescued") + ' · ' +
      paired.regressed + ' ' + t("ca phát sinh bỏ sót", "regressed") + '</p>';
    const record = research.cases.find(item => item.row_id === Number(caseInput.value));
    if (!record) return;
    const caseRows = methods.map(method => {
      const result = record.methods[method];
      return [label(method), result.score.toFixed(4), result.daily_rank, result.cutoff.toFixed(4), decisions[result.decision]];
    });
    $("research-case-view").innerHTML =
      pair(t("Input · giao dịch", "Input · transaction"),
        {row_id: record.row_id, time_seconds: record.time_seconds, amount: record.amount},
        t("Output · nhãn xác nhận khi chấm", "Output · historical evaluation label"),
        {fraud: record.label, hours: [record.batch * 6, (record.batch + 1) * 6]}) +
      rows(t("Điểm → ngưỡng → quyết định replay", "Score → cutoff → arrival decision"),
        [t("Phương pháp", "Method"), t("Điểm", "Score"), t("Hạng batch", "Batch rank"),
          t("Ngưỡng", "Cutoff"), t("Quyết định", "Decision")], caseRows) +
      '<details><summary>' + t("Xem đủ 29 feature", "Inspect all 29 features") +
      '</summary><pre>' + dump(record.features) + '</pre></details>' +
      '<details><summary>' + t("Feature lệch khỏi trung bình train", "Feature context relative to training data") + '</summary>' +
      '<p class="muted">' + t(
        "Năm feature có độ lệch chuẩn hóa lớn nhất. Đây là ngữ cảnh để điều tra, chưa xác định nguyên nhân của điểm model.",
        "Five features with the largest absolute standardized deviations from training means. These describe the input and leave score attribution unresolved.") + '</p>' +
      rows(t("So với train", "Compared with training data"),
        ["Feature", t("Giá trị", "Value"), t("Trung bình train", "Train mean"), t("Độ lệch train", "Train std.")],
        record.feature_context.map(item => [item.feature, item.value.toFixed(3), item.train_mean.toFixed(3), item.train_std.toFixed(3)])) + '</details>';
  }
  function chooseGroup() {
    caseInput.replaceChildren();
    const ids = research.case_groups[groupInput.value];
    options(caseInput, ids.map(id => [id, "Row " + id]), ids[0]);
    renderResearch();
  }
  seedInput.addEventListener("change", renderResearch);
  groupInput.addEventListener("change", chooseGroup);
  caseInput.addEventListener("change", renderResearch);
  chooseGroup();
  target.hidden = false;
}
