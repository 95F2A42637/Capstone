"""
Phase 8 — Part A+B: Final Project Structural Audit
Scans all expected checkpoints, outputs, and code files.
Produces outputs/final_project_audit.json and outputs/final_project_audit.md
"""

import os
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────────────────
# Expected file manifest
# ─────────────────────────────────────────────────────────
EXPECTED_FILES = {
    # ── Checkpoints ──────────────────────────────────────
    "models/dinov2_vits14_pretrain.pth": "DINOv2-Small ViT-S/14 backbone weights",
    "models/dinov2_authenticity_head.pth": "Trained MLP classification head (Phase 2)",
    "models/dinov2_frequency_fusion.pth": "DINOv2+FFT fusion model (Phase 3)",
    "models/calibration_params.json": "Temperature scaling parameter T (Phase 5)",
    "models/ood_params.json": "OOD centroids and cosine threshold (Phase 6)",
    # ── Source Code ─────────────────────────────────────
    "app/models_vfm.py": "DINOv2 backbone loader and classifier definitions",
    "app/explainability.py": "DINOv2PatchAttributor gradient-attribution module",
    "app/calibration.py": "Temperature scaling and entropy utilities",
    "app/frequency_features.py": "Compact 32-D FFT feature extractor",
    "app/ablation_frequency.py": "Phase 3 frequency ablation runner",
    "app/run_calibration.py": "Phase 5 calibration runner",
    "app/error_analysis.py": "Phase 7 error/failure-mode analysis",
    "app/run_error_analysis.py": "Phase 7 error analysis runner",
    "app/robustness.py": "Phase 6 perturbation robustness module",
    "app/tta.py": "Phase 6 test-time augmentation module",
    "app/ood_detection.py": "Phase 6 OOD detection module",
    "app/run_phase6.py": "Phase 6 combined runner",
    "app/streamlit_app.py": "Final Streamlit demonstration interface (Phase 7)",
    "app/validate_streamlit_pipeline.py": "Offline pipeline validation script (Phase 7)",
    # ── Dataset ─────────────────────────────────────────
    "dataset/test/authentic": "Authentic test images directory",
    "dataset/test/ai_edited": "AI-edited test images directory",
    # ── Outputs — Phase 1 ───────────────────────────────
    "outputs/dinov2_phase1_verification.md": "Phase 1 feature extraction verification report",
    # ── Outputs — Phase 2 ───────────────────────────────
    "outputs/dinov2_phase2_training_report.md": "Phase 2 DINOv2 training report",
    "outputs/dinov2_test_report.txt": "Phase 2 test evaluation metrics",
    "outputs/dinov2_accuracy_curve.png": "Phase 2 accuracy training curve",
    "outputs/dinov2_loss_curve.png": "Phase 2 loss training curve",
    "outputs/dinov2_confusion_matrix.png": "Phase 2 confusion matrix",
    "outputs/dinov2_confusion_matrix.csv": "Phase 2 confusion matrix CSV",
    # ── Outputs — Phase 3 ───────────────────────────────
    "outputs/frequency_ablation.csv": "Phase 3 frequency ablation results",
    "outputs/frequency_ablation_report.md": "Phase 3 frequency ablation report",
    "outputs/ablation_roc_comparison.png": "Phase 3 ROC comparison chart",
    # ── Outputs — Phase 4 ───────────────────────────────
    "outputs/dinov2_phase4_explainability_report.md": "Phase 4 explainability report",
    "outputs/explainability": "Phase 4 patch attribution visualizations directory",
    # ── Outputs — Phase 5 ───────────────────────────────
    "outputs/dinov2_phase5_calibration_report.md": "Phase 5 calibration report",
    "outputs/calibration": "Phase 5 calibration reliability diagrams directory",
    # ── Outputs — Phase 6 ───────────────────────────────
    "outputs/robustness_results.csv": "Phase 6A robustness results",
    "outputs/robustness_report.md": "Phase 6A robustness report",
    "outputs/robustness_comparison.png": "Phase 6A robustness comparison chart",
    "outputs/tta_results.csv": "Phase 6B TTA results",
    "outputs/tta_report.md": "Phase 6B TTA report",
    "outputs/ood_results.csv": "Phase 6C OOD results",
    "outputs/ood_report.md": "Phase 6C OOD report",
    "outputs/dinov2_phase6_robustness_ood_report.md": "Phase 6 combined master report",
    # ── Outputs — Phase 7 ───────────────────────────────
    "outputs/error_analysis_results.csv": "Phase 7 per-sample error analysis (N=2000)",
    "outputs/error_analysis_report.md": "Phase 7 error analysis report",
    "outputs/error_confusion_matrix.png": "Phase 7 semantic confusion matrix",
    "outputs/error_confidence_vs_correctness.png": "Phase 7 confidence/entropy distribution",
    "outputs/failure_mode_summary.md": "Phase 7 failure mode taxonomy",
    "outputs/dinov2_phase7_error_analysis_streamlit_report.md": "Phase 7 master report",
    # ── Miscellaneous ────────────────────────────────────
    "requirements.txt": "Python package requirements file",
    "README.md": "Project README",
}

# ─────────────────────────────────────────────────────────
# Part B — ResNet dependency audit in streamlit_app.py
# ─────────────────────────────────────────────────────────
STREAMLIT_PATH = os.path.join(PROJECT_ROOT, "app", "streamlit_app.py")
RESNET_KEYWORDS = ["resnet", "ResNet", "torchvision.models", "BasicBlock", "Bottleneck",
                   "ResidualBlock", "from app.train_model", "import train_model"]


def audit_resnet_dependency():
    """Return True if any ResNet keyword is found in streamlit_app.py."""
    if not os.path.exists(STREAMLIT_PATH):
        return None, "streamlit_app.py not found"
    with open(STREAMLIT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    hits = [kw for kw in RESNET_KEYWORDS if kw in content]
    return hits, content


def get_file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def run_audit():
    results = {}
    present = []
    missing = []

    for rel_path, description in EXPECTED_FILES.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        exists = os.path.exists(full_path)
        size = get_file_size(full_path) if exists and os.path.isfile(full_path) else None
        entry = {
            "path": rel_path,
            "description": description,
            "exists": exists,
            "size_bytes": size,
        }
        results[rel_path] = entry
        if exists:
            present.append(rel_path)
        else:
            missing.append(rel_path)

    # ResNet audit
    resnet_hits, _ = audit_resnet_dependency()
    resnet_clean = (resnet_hits is not None and len(resnet_hits) == 0)

    summary = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_expected": len(EXPECTED_FILES),
        "total_present": len(present),
        "total_missing": len(missing),
        "all_critical_present": len(missing) == 0,
        "resnet_dependency_in_streamlit": not resnet_clean,
        "resnet_hits": resnet_hits if resnet_hits else [],
    }

    return summary, results, missing


def write_outputs(summary, results, missing):
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── JSON ─────────────────────────────────────────────
    json_path = os.path.join(out_dir, "final_project_audit.json")
    payload = {"summary": summary, "files": results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[AUDIT] Written: {json_path}")

    # ── Markdown ─────────────────────────────────────────
    md_path = os.path.join(out_dir, "final_project_audit.md")
    lines = []
    lines.append("# Phase 8 — Final Project Structural Audit\n")
    lines.append(f"**Generated:** {summary['audit_timestamp']}  \n")
    lines.append(f"**Project Root:** `{PROJECT_ROOT}`\n\n")
    lines.append("---\n\n")

    lines.append("## Summary\n\n")
    lines.append(f"| Metric | Value |\n")
    lines.append(f"|--------|-------|\n")
    lines.append(f"| Expected files / dirs | {summary['total_expected']} |\n")
    lines.append(f"| Present | {summary['total_present']} |\n")
    lines.append(f"| Missing | {summary['total_missing']} |\n")
    all_ok = "PASS" if summary['all_critical_present'] else "FAIL"
    lines.append(f"| All Critical Present | **{all_ok}** |\n")
    resnet_ok = "PASS (clean)" if not summary['resnet_dependency_in_streamlit'] else "FAIL (ResNet found)"
    lines.append(f"| ResNet in streamlit_app.py | **{resnet_ok}** |\n\n")

    lines.append("---\n\n")
    lines.append("## Part B — ResNet Dependency Audit\n\n")
    if not summary["resnet_dependency_in_streamlit"]:
        lines.append("**Result:** PASS — No ResNet keywords detected in `app/streamlit_app.py`.  \n")
        lines.append("The final Streamlit interface is DINOv2-only as required.\n\n")
    else:
        lines.append("**Result:** FAIL — ResNet keywords detected:\n\n")
        for h in summary["resnet_hits"]:
            lines.append(f"- `{h}`\n")
        lines.append("\n")

    lines.append("---\n\n")
    lines.append("## File Manifest\n\n")
    lines.append("| Status | Path | Description | Size |\n")
    lines.append("|--------|------|-------------|------|\n")
    for rel_path, entry in results.items():
        status = "PRESENT" if entry["exists"] else "**MISSING**"
        size_str = f"{entry['size_bytes']:,} B" if entry["size_bytes"] is not None else "—"
        lines.append(f"| {status} | `{rel_path}` | {entry['description']} | {size_str} |\n")

    if missing:
        lines.append("\n---\n\n")
        lines.append("## Missing Files\n\n")
        for m in missing:
            lines.append(f"- `{m}`\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[AUDIT] Written: {md_path}")

    return json_path, md_path


if __name__ == "__main__":
    print("[AUDIT] Running Phase 8 structural audit...")
    summary, results, missing = run_audit()
    json_path, md_path = write_outputs(summary, results, missing)

    print("\n[AUDIT] === SUMMARY ===")
    print(f"  Expected:      {summary['total_expected']}")
    print(f"  Present:       {summary['total_present']}")
    print(f"  Missing:       {summary['total_missing']}")
    print(f"  All Critical:  {'PASS' if summary['all_critical_present'] else 'FAIL'}")
    print(f"  ResNet Check:  {'PASS' if not summary['resnet_dependency_in_streamlit'] else 'FAIL'}")
    if missing:
        print("\n[AUDIT] Missing items:")
        for m in missing:
            print(f"  - {m}")
    print("\n[AUDIT] Done.")
