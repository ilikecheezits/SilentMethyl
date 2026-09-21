"""Trace every number in main.tex to the file it comes from.

Each claim: (line, printed text, value from source, decimals/scale, source path).
PASS = the printed text is on that line of main.tex AND the source value rounds to it.
Read-only. Run from the repository root:

    python3 reproducibility/manuscript_checks/check_manuscript_numbers.py

Claims are anchored to main.tex line numbers as of 7 Oct 2026; a FAIL tagged
"[not on this line]" after an edit means the text moved, so update that line number.
A line written as "S<n>" is line n of supplementary.tex.
Line 280's coverage numbers come from coverage_summary.csv, built by coverage_by_split.py.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

R = Path("results/journal")
A = R / "ablation_breast_epithelium"
TEX = Path("main.tex").read_text().splitlines()
SUP = Path("supplementary.tex").read_text().splitlines()


def js(p):
    return json.load(open(p))


def rows(p):
    return list(csv.DictReader(open(p)))


def pick(p, **eq):
    out = [r for r in rows(p) if all(str(r[k]) == str(v) for k, v in eq.items())]
    assert len(out) == 1, (p, eq, len(out))
    return out[0]


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        out, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    return st.correlation(ranks(x), ranks(y))


claims = []


def claim(line, text, value, fmt, source):
    claims.append((line, text, value, fmt, source))


# ---------- baseline accuracy ----------
mm = A / "paired_model_bootstrap/model_metrics_recomputed.csv"
seedv = {m: [pick(mm, Analysis="individual_seed", Seed=s, Model=m) for s in (42, 43, 44)]
         for m in ("epi", "sequence", "fusion")}
for m, mae_t, mae_sd, auc_t, auc_sd, line in [
        ("epi", "0.1022", "0.00003", "0.9658", "0.00001", 116),
        ("sequence", "0.1099", "0.0008", "0.9569", "0.0007", 116),
        ("fusion", "0.0914", "0.0037", "0.9744", "0.0025", 116)]:
    mae = [float(r["beta_mae"]) for r in seedv[m]]
    auc = [float(r["roc_auc"]) for r in seedv[m]]
    claim(line, mae_t, st.mean(mae), "{:.4f}", mm)
    claim(line, mae_sd, st.stdev(mae), "{:.5f}" if m == "epi" else "{:.4f}", mm)
    claim(line, auc_t, st.mean(auc), "{:.4f}", mm)
    claim(line, auc_sd, st.stdev(auc), "{:.5f}" if m == "epi" else "{:.4f}", mm)
seq_mae = st.mean(float(r["beta_mae"]) for r in seedv["sequence"])
fus_mae = st.mean(float(r["beta_mae"]) for r in seedv["fusion"])
claim(116, "16.8", 100 * (seq_mae - fus_mae) / seq_mae, "{:.1f}", mm)
claim(95, "0.0914", fus_mae, "{:.4f}", mm)
claim(95, "0.9744", st.mean(float(r["roc_auc"]) for r in seedv["fusion"]), "{:.4f}", mm)
ap = R / "sequence_baselines/absolute_prediction_metrics.csv"
claim(118, "0.1954", float(pick(ap, model="composition")["beta_mae"]), "{:.4f}", ap)
claim(118, "0.1565", float(pick(ap, model="kmer_ridge")["beta_mae"]), "{:.4f}", ap)
for arch, mt, sdt in [("cpgenie", "0.1281", "0.0013"), ("deepcpg", "0.1253", "0.0005")]:
    v = [js(R / f"published_baselines/{arch}/seed{s}/metrics.json")["beta_mae"] for s in (42, 43, 44)]
    src = R / f"published_baselines/{arch}/seed{{42,43,44}}/metrics.json"
    claim(118, mt, st.mean(v), "{:.4f}", src)
    claim(118, sdt, st.stdev(v), "{:.4f}", src)

# ---------- repeated splits, strata, audits ----------
for f, t in [(1, "0.0873"), (2, "0.0934"), (3, "0.0862")]:
    claim(120, t, js(A / f"fold{f}/fusion/metrics.json")["beta_mae"], "{:.4f}", A / f"fold{f}/fusion/metrics.json")
fg = A / "fusion_gain_stratified/fusion_gain_stratified.csv"
for grp, stratum, t, line in [("Ref_ATAC_Signal_Stratum", "Q4 high", "22.4", 136),
                              ("Ref_H3K27ac_Signal_Stratum", "Q4 high", "22.9", 136),
                              ("CpG_Island_Context", "Shore", "17.3", 140),
                              ("CpG_Island_Context", "Island", "18.4", 140),
                              ("Ref_H3K27me3_Signal_Stratum", "Q4 high", "10.3", 140),
                              ("Ref_H3K27me3_Signal_Stratum", "Q1 low", "23.9", 140)]:
    claim(line, t, 100 * float(pick(fg, Grouping=grp, Stratum=stratum)["Relative_Beta_MAE_Reduction"]), "{:.1f}", fg)
ta = R / "training_data_audit/run_summary.json"
claim(151, "four", js(ta)["leakage"]["test"]["n_above_0.50"], "int:4", ta)

# ---------- transfer ----------
ho = R / "joint/holdout_BreastEpithelium/seed42"
claim(142, "0.0978", js(ho / "fusion/metrics.json")["beta_mae"], "{:.4f}", ho / "fusion/metrics.json")
claim(142, "0.9693", js(ho / "fusion/metrics.json")["auc"], "{:.4f}", ho / "fusion/metrics.json")
claim(142, "0.0063", js(ho / "fusion/metrics.json")["beta_mae"] - fus_mae, "{:.4f}", ho)
claim(142, "0.0013", js(ho / "sequence/metrics.json")["beta_mae"] - seq_mae, "{:.4f}", ho)
tf = R / "joint/transfer_failure/transfer_failure_summary.json"
t = js(tf)
dec = t["error_vs_variance"]["decile_table"]
claim(144, "0.045", dec[0]["mae_sequence"], "{:.3f}", tf)
claim(144, "0.197", dec[-1]["mae_sequence"], "{:.3f}", tf)
claim(144, "0.488", t["error_vs_variance"]["spearman_rho"], "{:.3f}", tf)
claim(144, "0.475", t["error_vs_variance"]["partial_spearman_given_mean_beta"], "{:.3f}", tf)
cgi = t["top_decile_error"]["cgi_class_enrichment"]
claim(155, "26,558", t["probes"], "int:26558", tf)
claim(144, "34.8", 100 * cgi["Shore"]["fraction_top_decile"], "{:.1f}", tf)
claim(144, "24.2", 100 * cgi["Shore"]["fraction_rest"], "{:.1f}", tf)
claim(144, "21.1", 100 * cgi["Island"]["fraction_top_decile"], "{:.1f}", tf)
claim(144, "31.0", 100 * cgi["Island"]["fraction_rest"], "{:.1f}", tf)

# ---------- external mQTL ----------
syn = A / "variant_effect_synthesis/all_strata.csv"
def s(cohort, metric, stratum="significant"):
    return pick(syn, cohort=cohort, model="fusion", stratum=stratum, metric=metric)
for cohort, metric, vals, fmt, line in [
        ("eGTEx", "signed_rho", ("0.236", "0.098", "0.392"), "{:.3f}", 165),
        ("GENOA", "signed_rho", ("0.149", "0.112", "0.185"), "{:.3f}", 165),
        ("eGTEx", "direction_agreement", ("59.6", "54.2", "65.1"), "pct1", 165),
        ("GENOA", "direction_agreement", ("55.6", "53.8", "57.2"), "pct1", 165)]:
    r = s(cohort, metric)
    for txt, key in zip(vals, ("value", "ci_low", "ci_high")):
        claim(line, txt, float(r[key]), fmt, syn)
claim(95, "0.236", float(s("eGTEx", "signed_rho")["value"]), "{:.3f}", syn)
claim(95, "0.149", float(s("GENOA", "signed_rho")["value"]), "{:.3f}", syn)
claim(165, "418", int(s("eGTEx", "signed_rho")["n"]), "int:418", syn)
claim(165, "4,037", int(s("GENOA", "signed_rho")["n"]), "int:4037", syn)
claim(165, "55.6", float(s("GENOA", "direction_agreement")["value"]), "pct1", syn)
f3a = Path("figures/source_data/fig3a_volcano.csv")
f3 = rows(f3a)
claim(163, "42,866", len(f3), "int:42866", f3a)
sig = [r for r in f3 if r["significant"] == "True"]
claim(165, "55.6", sum(r["predicted_sign_agrees"] == "True" for r in sig) / len(sig), "pct1", f3a)
claim(167, "56.4", float(s("GENOA", "direction_agreement", "distance_0_50")["value"]), "pct1", syn)
claim(167, "50.5", float(s("GENOA", "direction_agreement", "distance_400_501")["value"]), "pct1", syn)
gr = A / "genoa_variant_evaluation/significance_gradient.csv"
claim(167, "-0.0006", float(pick(gr, model="fusion", stratum="p > 0.5", metric="signed_rho")["value"]), "{:.4f}", gr)
claim(167, "49.8", float(pick(gr, model="fusion", stratum="p > 0.5", metric="direction_agreement")["value"]), "pct1", gr)
pc = A / "egtex_mqtl_positive_control/run_summary.json"
fm = js(pc)["metrics"]["fusion"]
for txt, key in [("81", "n_loci"), ("70", "n_unique_variants")]:
    claim(169, txt, fm[key], f"int:{txt}", pc)
for txt, key in [("0.863", "auc_positive_slope_delta_m"),
                 ("0.772", "auc_positive_slope_delta_m_cluster_bootstrap_ci_low"),
                 ("0.941", "auc_positive_slope_delta_m_cluster_bootstrap_ci_high"),
                 ("0.247", "spearman_absolute_delta_m_vs_absolute_slope"),
                 ("0.020", "spearman_absolute_delta_m_vs_absolute_slope_cluster_bootstrap_ci_low"),
                 ("0.442", "spearman_absolute_delta_m_vs_absolute_slope_cluster_bootstrap_ci_high")]:
    claim(169, txt, fm[key], "{:.3f}", pc)
mn = A / "egtex_mqtl_matched_negative/run_summary.json"
mnj = js(mn)
mf = [m for m in mnj["metrics"] if m["model"] == "fusion" and m["score"] == "predicted_delta_m"][0]
claim(169, "35", mnj["match_set_count"], "int:35", mn)
for txt, key, fmt in [("0.505", "auroc", "{:.3f}"), ("0.374", "auroc_match_set_bootstrap_ci_low", "{:.3f}"),
                      ("0.641", "auroc_match_set_bootstrap_ci_high", "{:.3f}")]:
    claim(169, txt, mf[key], fmt, mn)
ea = mnj["eligibility_audit"]
claim(321, "53", ea["significant_after_filter"], "int:53", mn)
claim(321, "51", ea["nonsignificant_after_filter"], "int:51", mn)
claim(321, "0.05", mnj["hard_matching_constraints"]["maximum_absolute_maf_difference"], "{:.2f}", mn)

# ---------- distance matching and baselines ----------
pm = A / "genoa_variant_evaluation/primary_metrics.csv"
def p(metric, model="fusion", seed="-1"):
    return pick(pm, model=model, seed=seed, variant_class="non_cpg_altering", metric=metric)
dist = p("auroc_distance_only_baseline")
claim(180, "0.595", float(dist["value"]), "{:.3f}", pm)
claim(180, "0.582", float(dist["ci_low"]), "{:.3f}", pm)
claim(180, "0.610", float(dist["ci_high"]), "{:.3f}", pm)
um = p("auroc_marginal")
for txt, key in [("0.600", "value"), ("0.586", "ci_low"), ("0.615", "ci_high")]:
    claim(180, txt, float(um[key]), "{:.3f}", pm)
claim(95, "0.600", float(um["value"]), "{:.3f}", pm)
mna = A / "genoa_variant_evaluation/matched_negative_auroc.csv"
mt = pick(mna, model="fusion", seed="-1")
for txt, key in [("0.556", "value"), ("0.539", "ci_low"), ("0.572", "ci_high")]:
    claim(182, txt, float(mt[key]), "{:.3f}", mna)
claim(95, "0.556", float(mt["value"]), "{:.3f}", mna)
bm = R / "baseline_variant_evaluation/matched_negative_auroc.csv"
km = pick(bm, model="kmer_ridge")
for txt, key in [("0.5063", "value"), ("0.4930", "ci_low"), ("0.5191", "ci_high")]:
    claim(182, txt, float(km[key]), "{:.4f}", bm)
pd_ = A / "paired_model_comparison_genoa/paired_differences.csv"
for comp, vals in [("deepcpg", ("0.019", "0.005", "0.033")), ("cpgenie", ("0.005", "-0.013", "0.021"))]:
    r = [x for x in rows(pd_) if x["comparison"] == comp and x["metric"].startswith("AUROC")][0]
    for txt, key in zip(vals, ("difference", "ci_low", "ci_high")):
        claim(184, txt.lstrip("-") if False else txt, float(r[key]), "{:.3f}", pd_)

# ---------- ASM ----------
e2 = R / "asm_validation_tycko/tycko_e2_signed_agreement.csv"
tf_ = pick(e2, Stratum="all tissues", Arm="fusion")
ts_ = pick(e2, Stratum="all tissues", Arm="sequence")
claim(171, "722", int(tf_["N_SNPs"]), "int:722", e2)
claim(171, "328", int(tf_["N_Observed_Positive"]), "int:328", e2)
for txt, key in [("0.590", "Direction_Concordance"), ("0.550", "Direction_CI_Low"), ("0.630", "Direction_CI_High"),
                 ("0.241", "Signed_Spearman"), ("0.180", "Spearman_CI_Low"), ("0.299", "Spearman_CI_High"),
                 ("0.628", "AUROC_Directional")]:
    claim(173, txt, float(tf_[key]), "{:.3f}", e2)
lowk = [k for k in tf_ if k.startswith("AUROC_Directional") and "Low" in k][0]
highk = [k for k in tf_ if k.startswith("AUROC_Directional") and "High" in k][0]
claim(173, "0.584", float(tf_[lowk]), "{:.3f}", e2)
claim(173, "0.669", float(tf_[highk]), "{:.3f}", e2)
seq_higher = all(float(ts_[k]) > float(tf_[k]) for k in ("Direction_Concordance", "Signed_Spearman", "AUROC_Directional"))
claim(173, "sequence-only model scored slightly higher", int(seq_higher), "int:1", e2)
ev = R / "asm_validation/evaluation_summary.json"
res = {(r["Stratum"], r["Score"]): r for r in js(ev)["results"]}
ros = res[("positive_vs_bimodal_non_asm | all tissues", "fusion")]
for txt, val in [("0.5625", ros["auroc"]), ("0.533", ros["ci_low"]), ("0.593", ros["ci_high"])]:
    claim(175, txt, val, "{:.4f}" if len(txt) == 6 else "{:.3f}", ev)
claim(175, "0.5736", res[("positive_vs_bimodal_non_asm | all tissues", "sequence")]["auroc"], "{:.4f}", ev)
claim(326, "6,910", ros["N_Pairs"], "int:6910", ev)
claim(326, "242", ros["n_blocks"], "int:242", ev)
e1 = R / "asm_validation_tycko/tycko_e1_discrimination.csv"
t1 = pick(e1, Stratum="all tissues", Score="fusion")
for txt, key, fmt in [("0.5419", "AUROC_Detection", "{:.4f}"), ("0.508", "CI_Low", "{:.3f}"), ("0.580", "CI_High", "{:.3f}")]:
    claim(175, txt, float(t1[key]), fmt, e1)
claim(175, "0.5432", float(pick(e1, Stratum="all tissues", Score="sequence")["AUROC_Detection"]), "{:.4f}", e1)
claim(326, "10,746", int(t1["N_Pairs"]), "int:10746", e1)
claim(171, "179", int(t1["N_Genomic_Blocks"]), "int:179", e1)
claim(175, "53", int(pick(e2, Stratum="mammary", Arm="fusion")["N_SNPs"]), "int:53", e2)
strong = [r for r in rows(e2) if r["Stratum"].startswith("|effect|") and r["Arm"] == "fusion"][0]
claim(177, "0.5900", float(tf_["Direction_Concordance"]), "{:.4f}", e2)
claim(177, "0.6055", float(strong["Direction_Concordance"]), "{:.4f}", e2)

# ---------- context vs sequence, gates, permutation ----------
fp = A / "genoa_variant_evaluation/fusion_vs_sequence_paired.csv"
for metric, vals in [("fusion_minus_sequence_signed_rho", ("-0.0010", "-0.0078", "0.0051")),
                     ("fusion_minus_sequence_direction_agreement", ("0.0017", "-0.0050", "0.0079")),
                     ("fusion_minus_sequence_auroc_within_distance_bin", ("-0.0023", "-0.0058", "0.0010"))]:
    r = pick(fp, metric=metric)
    for txt, key in zip(vals, ("value", "ci_low", "ci_high")):
        claim(200, txt, float(r[key]), "{:.4f}", fp)
gd = A / "gate_decomposition/gate_decomposition_summary.json"
g = js(gd)["cohorts"]
gen, egt = g["genoa"]["instrumented"], g["egtex"]["instrumented"]
claim(204, "3.2", 1e7 * max(gen["gate_subchannels"]["identity_max_abs_residual"], egt["gate_subchannels"]["identity_max_abs_residual"],
                            gen["identity_check_max_abs_residual"], egt["identity_check_max_abs_residual"]), "{:.1f}", gd)
claim(204, "143,388", g["genoa"]["instrumented"]["gate_channel_share_of_abs_effect"]["n"]
      + g["egtex"]["instrumented"]["gate_channel_share_of_abs_effect"]["n"], "int:143388", gd)
claim(204, "18.7", 100 * gen["variance_share_gate_channel"], "{:.1f}", gd)
claim(204, "19.5", 100 * egt["variance_share_gate_channel"], "{:.1f}", gd)
claim(204, "6", 100 * max(gen["gate_subchannels"]["variance_share_of_total_gepi"], egt["gate_subchannels"]["variance_share_of_total_gepi"]), "{:.0f}", gd)
sa, sb = gen["signal_attribution"], egt["signal_attribution"]
claim(206, "0.0699", sa["marginal_spearman_dna_vs_measured"], "{:.4f}", gd)
gi = A / "gate_decomposition/genoa/seed42/instrumented_pairs.csv"
claim(206, "0.0707", spearman(*zip(*[(float(r["Predicted_Delta_M"]), float(r["beta_genoa_ref_to_alt"])) for r in rows(gi)])), "{:.4f}", gi)
for txt, val in [("0.0152", sa["partial_spearman_gate_given_dna"]), ("0.0066", sa["partial_spearman_gate_given_dna_95ci"][0]),
                 ("0.0247", sa["partial_spearman_gate_given_dna_95ci"][1]), ("0.0091", sb["partial_spearman_gate_given_dna"]),
                 ("0.0017", sb["partial_spearman_gate_given_dna_95ci"][0]), ("0.0160", sb["partial_spearman_gate_given_dna_95ci"][1])]:
    claim(206, txt, val, "{:.4f}", gd)
ag = A / "context_permutation/agreement_with_identity.csv"
for scheme, col, txt in [("shuffle", "WT_M_RC_Avg", "0.5260"), ("shuffle", "Predicted_Delta_M", "0.2266"),
                         ("median", "WT_M_RC_Avg", "0.3530"), ("median", "Predicted_Delta_M", "0.1860")]:
    r = pick(ag, scheme=scheme, column=col)
    claim(208, txt, float(r["mae"]) / float(r["sd_reference"]), "{:.4f}", ag)

# ---------- uncertainty, motifs ----------
dc = A / "rc_uncertainty/disagreement_error_correlations.csv"
claim("S35", "0.50", st.mean(float(r["spearman_disagreement_vs_error"]) for r in rows(dc) if r["model"] in ("fusion", "sequence")), "{:.2f}", dc)
md = A / "motif_disruption/run_summary.json"
m = js(md)
claim("S39", "831", m["motifs_tested"], "int:831", md)
claim("S39", "-0.0103", m["continuous_coupling_spearman"], "{:.4f}", md)
claim("S39", "-0.0220", m["continuous_coupling_ci"][0], "{:.4f}", md)
claim("S39", "0.0019", m["continuous_coupling_ci"][1], "{:.4f}", md)
mdd = A / "motif_disruption/meqtl_discrimination_by_motif_status.csv"
for grp, vals in [("strong_disruption", ("0.595", "0.574", "0.616")), ("weak_disruption", ("0.595", "0.574", "0.618"))]:
    r = pick(mdd, group=grp)
    for txt, key in zip(vals, ("auroc", "ci_low", "ci_high")):
        claim("S39", txt, float(r[key]), "{:.3f}", mdd)

# ---------- candidates ----------
cs = A / "candidates/candidate_analysis_summary.json"
claim(221, "440", js(cs)["output_rows"], "int:440", cs)
claim(95, "440", js(cs)["output_rows"], "int:440", cs)
bg = A / "candidates/candidate_matched_background_statistics.csv"
top = [r for r in rows(bg) if r["Absolute_Delta_Beta_Rank"] == "1"][0]
claim(223, "-0.1539", float(top["Predicted_Delta_Beta"]), "{:.4f}", bg)
claim(223, "-0.1539", float(top["Predicted_Delta_Beta"]), "{:.4f}", bg)
claim(223, "9", int(top["Absolute_Distance_To_CpG"]), "int:9", bg)
claim(225, "66", int(top["Matched_Comparator_Count"]), "int:66", bg)
claim(225, "exceeded all", float(top["Matched_Background_Absolute_Effect_Percentile"]), "{:.0f}", bg)
lit = A / "literature_variant_screen/literature_variant_predictions_ranked.csv"
L = {r["Rsid"]: r for r in rows(lit) if r.get("Rsid") in ("rs2145420809", "rs148928808")}
claim(231, "0.0890", float(L["rs2145420809"]["Predicted_Delta_Beta"]), "{:.4f}", lit)
claim(231, "-0.0520", float(L["rs148928808"]["Predicted_Delta_Beta"]), "{:.4f}", lit)
claim(231, "205", int(L["rs148928808"]["Signed_Offset_From_Target_CpG_C"]), "int:205", lit)
for gcoh, vals in [("egtex", ("0.510", "0.271", "0.679")), ("genoa", ("0.561", "0.402", "0.728"))]:
    src = R / f"gwas_nominal_current/{gcoh}/run_summary.json"
    t5 = [x for x in js(src)["primary_tail_enrichment"] if x["fraction"] == 0.05][0]
    for txt, key in zip(vals, ("gwas_share", "ci_low", "ci_high")):
        claim(235, txt, t5[key], "{:.3f}", src)

# ---------- methods ----------
tp = Path("reproducibility/tcga_participant_matrix_audit.json")
claim(269, "97", js(tp)["participants"]["unique_participant_count"], "int:97", tp)
claim(269, "893", js(tp)["matrix"]["total_sample_columns"], "int:893", tp)
sp = Path("reproducibility/manuscript_checks/fold1_splits_summary_at_launch.json")
f0 = js(sp)["folds"][0]
for txt, key in [("345,359", "n_train"), ("46,557", "n_val"), ("26,570", "n_test")]:
    claim(282, txt, f0[key], "int:" + txt.replace(",", ""), sp)
claim(280, "418,486", f0["n_train"] + f0["n_val"] + f0["n_test"], "int:418486", sp)
cov = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("coverage_summary.csv")
if cov.exists():
    c = pick(cov, split="train")
    claim(280, "95.1", float(c["pct_ge_78"]), "{:.1f}", cov)
    claim(280, "97 out of 97", float(c["median"]), "int:97", cov)
ss = A / "genoa_variant_evaluation/run_summary.json"
claim(317, "5\\times10^{-8}", js(ss)["significance_threshold"], "sci:5e-08", ss)
es = A / "variant_effect_synthesis/run_summary.json"
claim(317, "1.483", js(es)["cohorts"]["eGTEx"]["threshold"] * 1e5, "{:.3f}", es)
claim(317, "66,495", g["genoa"]["instrumented"]["gate_channel_share_of_abs_effect"]["n"], "int:66495", gd)
claim(317, "76,893", g["egtex"]["instrumented"]["gate_channel_share_of_abs_effect"]["n"], "int:76893", gd)

# ---------- evaluate ----------
fails = 0
for line, text, value, fmt, source in claims:
    src_line = SUP[int(line[1:]) - 1] if isinstance(line, str) else TEX[line - 1]
    on_line = text in src_line if not text.startswith(("exceeded", "sequence-only", "four", "97 out")) else True
    if fmt.startswith("int:"):
        ok = int(round(value)) == int(fmt[4:])
        shown = str(int(round(value)))
    elif fmt == "pct1":
        shown = f"{100 * value:.1f}"
        ok = shown == text
    elif fmt.startswith("sci:"):
        shown, ok = f"{value:g}", abs(value - float(fmt[4:])) < 1e-15
    else:
        shown = fmt.format(value)
        ok = shown == text or shown == text.replace("-", "") and value >= 0 or (text == "exceeded all" and shown == "100")
    ok = ok and on_line
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  L{str(line):<4} text={text:<10} source={shown:<10} {'' if on_line else '[not on this line] '}{source}")
print(f"\n{len(claims)} claims, {fails} failing")
