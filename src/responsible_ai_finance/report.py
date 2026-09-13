"""AuditReport: bundles fairness, explainability and robustness results and
renders them as Markdown (for a PR/ticket) or a plain dict (for logging /
storing alongside a model version in a registry)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .explainability import ExplainabilityResult
from .fairness import FairnessResult
from .robustness import RobustnessResult


@dataclass
class AuditReport:
    model_name: str
    fairness: FairnessResult
    explainability: ExplainabilityResult
    robustness: RobustnessResult
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "generated_at": self.generated_at,
            "fairness": self.fairness.to_dict(),
            "explainability": self.explainability.to_dict(),
            "robustness": self.robustness.to_dict(),
        }

    def flags(self) -> list[str]:
        """Return a short list of governance flags a reviewer should look at first."""
        flags: list[str] = []
        if not self.fairness.passes_80_percent_rule:
            flags.append(
                f"Disparate impact ratio {self.fairness.disparate_impact_ratio:.2f} is below the "
                "0.80 threshold (80% rule) — review group-level selection rates."
            )
        if self.robustness.noise_flip_rate > 0.10:
            flags.append(
                f"{self.robustness.noise_flip_rate:.1%} of predictions flip under 5% input noise — "
                "model may be brittle near decision boundaries."
            )
        if self.robustness.most_fragile_feature:
            rate = self.robustness.feature_dropout_flip_rate[self.robustness.most_fragile_feature]
            if rate > 0.10:
                flags.append(
                    f"Zeroing '{self.robustness.most_fragile_feature}' flips {rate:.1%} of predictions — "
                    "check upstream data-quality guarantees for this feature."
                )
        if self.explainability.local_fidelity_mae > 0.10:
            flags.append(
                f"SHAP local-fidelity MAE is {self.explainability.local_fidelity_mae:.3f} — "
                "additive attribution may not be fully faithful to the model's output."
            )
        return flags

    def to_markdown(self) -> str:
        lines = [
            f"# Responsible AI Audit — {self.model_name}",
            f"_Generated {self.generated_at}_",
            "",
            "## Governance Flags",
        ]
        flags = self.flags()
        lines += [f"- ⚠️ {f}" for f in flags] if flags else ["- ✅ No threshold breaches detected."]

        lines += ["", "## Fairness", ""]
        f = self.fairness
        lines.append(f"- Demographic parity difference: **{f.demographic_parity_difference:.4f}**")
        lines.append(f"- Disparate impact ratio: **{f.disparate_impact_ratio:.4f}** (80% rule: {'PASS' if f.passes_80_percent_rule else 'FAIL'})")
        if f.equalized_odds_tpr_difference is not None:
            lines.append(f"- Equalized odds — TPR difference: **{f.equalized_odds_tpr_difference:.4f}**")
        if f.equalized_odds_fpr_difference is not None:
            lines.append(f"- Equalized odds — FPR difference: **{f.equalized_odds_fpr_difference:.4f}**")
        lines.append("")
        lines.append("| Group | n | Selection rate | TPR | FPR |")
        lines.append("|---|---|---|---|---|")
        for g in f.group_metrics:
            tpr = f"{g.tpr:.3f}" if g.tpr is not None else "n/a"
            fpr = f"{g.fpr:.3f}" if g.fpr is not None else "n/a"
            lines.append(f"| {g.group} | {g.n} | {g.selection_rate:.3f} | {tpr} | {fpr} |")

        lines += ["", "## Explainability", ""]
        e = self.explainability
        lines.append(f"- Top features by mean |SHAP value|: {', '.join(e.top_features[:5])}")
        lines.append(f"- SHAP local-fidelity MAE: **{e.local_fidelity_mae:.4f}**")

        lines += ["", "## Robustness", ""]
        r = self.robustness
        lines.append(f"- Prediction flip rate under 5% Gaussian noise: **{r.noise_flip_rate:.1%}**")
        if r.most_fragile_feature:
            lines.append(
                f"- Most fragile feature to dropout: **{r.most_fragile_feature}** "
                f"({r.feature_dropout_flip_rate[r.most_fragile_feature]:.1%} flip rate)"
            )

        return "\n".join(lines)
