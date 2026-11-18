/* Label-informed IF ablation with fixed development selection. */
if (data.feature_study) {
  const study = data.feature_study, target = $("feature-study");
  const configs = study.selection.candidates;
  const selected = configs.find(config => config.name === study.selection.standalone);
  const secondary = configs.find(config => config.name === study.selection.secondary);
  function configLabel(config) {
    return config.features.length + " " + t("feature", "features") + " · " +
      (config.normal_only ? t("train hợp lệ", "normal-only training") : t("toàn bộ train", "all training rows"));
  }
  const anchor = document.createElement("a"), jump = document.createElement("p");
  anchor.href = "#feature-study";
  anchor.textContent = t("IF học đúng feature có khá hơn không? ↓", "Does a better feature set improve IF? ↓");
  jump.className = "jump";
  jump.append(anchor);
  document.querySelector("header").append(jump);
  const steps = t(["Tập train","Feature được chọn","Điểm model","Queue review","Kết quả"],
                  ["Training data","Selected features","Model scores","Review queue","Outcome"]);
  let step = 0;
  target.innerHTML = '<p class="kicker">' + t("NGHIÊN CỨU / CÓ DÙNG NHÃN TRAIN", "RESEARCH / TRAIN LABELS INFORM THE DESIGN") + '</p>' +
    '<h2>' + t("IF học đúng feature có khá hơn không?", "Does a better feature set improve IF?") + '</h2>' +
    '<p class="muted">' + t(
      "So 8 thiết kế trên 3 seed. Feature được chọn bằng nhãn trong train. Normal-only chỉ học các dòng train đã xác nhận hợp lệ. Lựa chọn chốt trên dev.",
      "Eight designs across three seeds. Feature relevance is learned using training labels. Normal-only forests fit confirmed legitimate training rows. Choices are frozen on dev.") + '</p>' +
    '<p class="flag">' + t("IF độc lập", "Standalone IF") + ' · ' + configLabel(selected) + '</p>' +
    '<p class="muted">' + t("IF bổ sung cho XGB", "IF alongside XGB") + ' · ' + configLabel(secondary) +
    ' · ' + t("Chính sách chọn trên dev", "Policy selected on dev") + ': ' +
    (study.selection.policy === "main_50" ? "50 main + 0 IF" : "45 main + 5 IF") + '</p>' +
    '<div class="controls jump"><label>Seed <select id="feature-seed"></select></label>' +
    '<label>' + t("Loại ca", "Case group") + ' <select id="feature-group"></select></label>' +
    '<label>' + t("Giao dịch", "Transaction") + ' <select id="feature-case"></select></label></div>' +
    '<div id="feature-summary" aria-live="polite"></div>' +
    '<nav id="feature-flow" class="flow" aria-label="' + t("Flow IF", "IF study flow") + '"></nav>' +
    '<div id="feature-trace" aria-live="polite"></div>' +
    '<details><summary>' + t("Mọi thiết kế đã thử", "Every design tried") + '</summary><div id="feature-ablation"></div></details>' +
    '<details><summary>' + t("Thứ hạng feature từ train", "Feature relevance from training data") + '</summary>' +
    rows(t("Mutual information dùng nhãn train", "Mutual information uses training labels"),
      ["Feature", "MI"], study.selection.train_feature_relevance.map(row => [row.feature, row.mutual_information.toFixed(6)])) + '</details>';
  const seedInput = $("feature-seed"), groupInput = $("feature-group"), caseInput = $("feature-case");
  options(seedInput, study.seeds.map(seed => [seed, seed]), 42);
  const buttons = steps.map((name, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = (index + 1) + " · " + name;
    button.addEventListener("click", () => { step = index; renderFeature(); });
    $("feature-flow").append(button);
    return button;
  });
  const groups = [
    ["if_unique_catches", t("IF tìm thêm ngoài XGB", "Unique IF catches beyond XGB")],
    ["standalone_rescued", t("IF độc lập được cứu", "Standalone IF rescued")],
    ["standalone_regressed", t("IF độc lập phát sinh bỏ sót", "Standalone IF regressed")],
    ["reserved_rescued", t("45 + 5 được cứu", "45 + 5 rescued")],
    ["reserved_regressed", t("45 + 5 phát sinh bỏ sót", "45 + 5 regressed")],
    ["if_false_alerts", t("IF cảnh báo nhầm", "IF false alerts")]
  ];
  function methodLabel(key) {
    if (key.startsWith("all_top29")) return t("IF 29 feature", "IF · all 29 features");
    if (key.startsWith("main_50")) return "XGB · 50";
    if (key.startsWith("reserved")) return "XGB + IF · 45 + 5";
    return t("IF chọn trên dev", "Dev-selected IF");
  }
  const decisionLabel = key => ({
    review: t("Duyệt", "Review"), below_cutoffs: t("Dưới ngưỡng", "Below cutoffs"),
    capacity_full: t("Quota đã đầy", "Queue cap reached"), invalid_data: t("Dữ liệu lỗi", "Invalid data")
  })[key];
  function renderFeature() {
    const seed = seedInput.value;
    const keys = ["all_top29_" + seed, selected.name + "_" + seed, "main_50_" + seed, "reserved_45_5_" + seed];
    const uniqueKeys = [...new Set(keys)];
    $("feature-summary").innerHTML = rows(t("Cùng trần 100 lượt duyệt trên 12 giờ", "Same ceiling of 100 reviews over 12 hours"),
      [t("Phương án", "Method"), t("Bắt được", "Caught"), t("Cảnh báo nhầm", "False alerts"),
        t("Đã duyệt", "Reviewed"), t("IF bổ sung bắt được", "Caught via IF branch")],
      uniqueKeys.map(key => {
        const result = study.results[key].standalone || study.results[key].arrival;
        return [methodLabel(key), result.caught + " / 115", result.false_alerts, result.reviewed,
          key.startsWith("reserved") ? result.branches.if.caught : "—"];
      }));
    $("feature-ablation").innerHTML = rows(t("Cấu hình và kết quả từng seed", "Configurations and results at the selected seed"),
      [t("Thiết kế IF", "IF design"), t("Dev độc lập", "Dev standalone"),
        t("Đánh giá độc lập", "Evaluation standalone"), t("Cảnh báo nhầm", "False alerts"), "AP"],
      configs.map(config => {
        const dev = config.dev_runs.find(run => run.seed === Number(seed));
        const result = study.results[config.name + "_" + seed];
        return [configLabel(config), dev.standalone.caught + " / 96",
          result.standalone.caught + " / 115", result.standalone.false_alerts, percent(result.batch.ap)];
      }));
    buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === step)));
    const record = study.cases.find(row => row.row_id === Number(caseInput.value));
    if (!record) {
      $("feature-trace").innerHTML = '<p class="flag">' + t("Không có ca trong nhóm này.", "No cases in this group.") + '</p>';
      return;
    }
    let content = '';
    if (step === 0) {
      content = pair(t("IF gốc · tập train", "Original IF · training set"),
        {rows: study.train_rows, features: 29},
        t("IF chọn trên dev · tập train", "Selected IF · training set"),
        {rows: selected.training_rows, features: selected.features.length, normal_only: selected.normal_only});
    } else if (step === 1) {
      content = pair(t("Feature IF độc lập", "Standalone IF features"), selected.features,
        t("Feature IF bổ sung", "Secondary IF features"), secondary.features);
      content += '<details><summary>' + t("Giá trị của giao dịch", "Transaction feature values") +
        '</summary><pre>' + dump(record.features) + '</pre></details>';
    } else if (step === 2) {
      content = rows(t("Điểm và cutoff chốt trên dev", "Scores and dev-frozen cutoffs"),
        [t("Phương án", "Method"), t("Điểm chính", "Main score"), t("Điểm IF bổ sung", "Secondary IF score"), t("Ngưỡng chính", "Main cutoff"), "IF cutoff"],
        uniqueKeys.map(key => {
          const trace = record.traces[key], result = study.results[key];
          const cutoffs = result.cutoffs || [result.cutoff, null];
          return [methodLabel(key), trace.main_score.toFixed(4), cutoffs[1] === null ? "—" : trace.if_score.toFixed(4),
            cutoffs[0].toFixed(4), cutoffs[1] === null ? "—" : cutoffs[1].toFixed(4)];
        }));
    } else if (step === 3) {
      content = rows(t("Quota và quyết định lúc đến", "Quota and decision on arrival"),
        [t("Phương án", "Method"), t("Nhánh", "Branch"), t("Slot trước", "Slots before"),
          t("Slot sau", "Slots after"), t("Quyết định", "Decision")],
        uniqueKeys.map(key => {
          const trace = record.traces[key];
          return [methodLabel(key), trace.branch, trace.main_before + " / " + trace.if_before,
            trace.main_after + " / " + trace.if_after, decisionLabel(trace.decision)];
        }));
    } else {
      content = pair(t("Input · giao dịch", "Input · transaction"), {row_id: record.row_id, amount: record.amount},
        t("Nhãn hậu kiểm", "Historical label"), {fraud: record.label});
      content += rows(t("Đối chiếu quyết định", "Compare decisions"),
        [t("Phương án", "Method"), t("Quyết định", "Decision")],
        uniqueKeys.map(key => [methodLabel(key), decisionLabel(record.traces[key].decision)]));
    }
    $("feature-trace").innerHTML = '<p class="kicker">ROW ' + record.row_id + ' / ' + steps[step] + '</p>' + content;
  }
  function chooseCase() {
    caseInput.replaceChildren();
    const ids = study.case_groups[groupInput.value];
    options(caseInput, ids.map(id => [id, "Row " + id]), ids[0]);
    renderFeature();
  }
  function chooseSeed() {
    groupInput.replaceChildren();
    const choices = groups.map(([key, name]) => {
      const group = key + "_" + seedInput.value;
      return [group, name + " (" + study.case_groups[group].length + " " + t("ca mẫu", "examples") + ")"];
    });
    options(groupInput, choices, choices.find(([key]) => study.case_groups[key].length)?.[0] || choices[0][0]);
    chooseCase();
  }
  seedInput.addEventListener("change", chooseSeed);
  groupInput.addEventListener("change", chooseCase);
  caseInput.addEventListener("change", renderFeature);
  chooseSeed();
  target.hidden = false;
}
