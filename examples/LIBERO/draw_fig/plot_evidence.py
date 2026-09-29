"""Plot chunk token counts and selected-token evidence from rollout JSONL files."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_trace(source: Path, output_dir: Path, axes=None, show_legend=True) -> None:
    chunks, totals, low_counts, ratios, token_x, token_evidence = [], [], [], [], [], []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            chunk = int(record.get("chunk_idx", len(chunks)))
            evidence = np.asarray(record["action_token_evidence"], dtype=float)
            if evidence.ndim != 1 or not evidence.size or not np.all(np.isfinite(evidence)):
                raise ValueError(f"{source}:{line_number}: expected finite, nonempty evidence array")
            count = int(np.count_nonzero(evidence < 4.0))
            chunks.append(chunk)
            totals.append(evidence.size)
            low_counts.append(count)
            ratios.append(count / evidence.size)
            token_x.extend(chunk + (np.arange(evidence.size) + 0.5) / evidence.size)
            token_evidence.extend(evidence)
    if not chunks:
        raise ValueError(f"Empty trace: {source}")

    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        standalone = axes is None
        if standalone:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        counts_ax, evidence_ax = axes
        fig = counts_ax.figure
        counts_ax.bar(chunks, totals, color="gray", alpha=0.35, label="Total action tokens")
        counts_ax.bar(chunks, low_counts, color="tab:red", alpha=0.55,
                      label="Low-evidence tokens")
        ratio_ax = counts_ax.twinx()
        ratio_ax.plot(chunks, ratios, color="black", marker="o", markersize=3,
                      linewidth=1.2, label="Low-evidence ratio")
        ratio_ax.set(ylabel="Low-evidence ratio", ylim=(0, 1))
        counts_ax.set_ylabel("Token count")
        counts_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        handles, labels = counts_ax.get_legend_handles_labels()
        extra_handles, extra_labels = ratio_ax.get_legend_handles_labels()
        if show_legend:
            counts_ax.legend(handles + extra_handles, labels + extra_labels,
                         loc="upper center", bbox_to_anchor=(0.5, -0.2),
                         ncol=1, fontsize=10, frameon=True, fancybox=False,
                         edgecolor="gray", columnspacing=1)

        # Display clipping only: counts and ratios above use the raw evidence.
        token_x = np.asarray(token_x)
        raw_evidence = np.asarray(token_evidence)
        display_evidence = np.clip(raw_evidence, 0, 8) / 4.0 - 1.0
        low_mask = raw_evidence < 4
        for mask, color, label in (
            (~low_mask, "tab:green", "High-evidence tokens"),
            (low_mask, "tab:red", "Low-evidence tokens"),
        ):
            evidence_ax.scatter(token_x[mask], display_evidence[mask], s=10,
                                alpha=0.75, color=color, clip_on=True, label=label)
        evidence_ax.axhline(0, color="tab:red", linestyle="--", linewidth=1.2,
                           label="Threshold = 0")
        evidence_ax.set(ylabel="Normalized evidence", ylim=(-1.1, 1.1),
                        yticks=[-1, -0.5, 0, 0.5, 1])
        if show_legend:
            evidence_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2),
                           ncol=1, fontsize=10, frameon=True, fancybox=False,
                           edgecolor="gray", columnspacing=1)
        for axis in (counts_ax, evidence_ax):
            axis.set_xlabel("Chunk index")
            axis.set_xlim(min(chunks) - 0.6, max(chunks) + 1)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(True, alpha=0.25)
            axis.set_axisbelow(True)
        if standalone:
            fig.tight_layout(w_pad=2.5)
            save_figure(fig, output_dir, f"{source.stem}_evidence")
    print(f"{source.name}: {len(chunks)} chunks, {sum(totals)} tokens, "
          f"{sum(low_counts)} tokens below 4")


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        destination = output_dir / f"{stem}.{extension}"
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        print(destination)
    plt.close(fig)


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="Rollout JSONL files; defaults to both traces beside this script")
    parser.add_argument("--output-dir", type=Path, default=directory / "outputs")
    args = parser.parse_args()
    sources = args.inputs or sorted(directory.glob("rollout_*.jsonl"))
    if not sources:
        parser.error("No rollout JSONL files found")
    if len(sources) != 2:
        parser.error("Expected two trajectories: one failure and one success")
    failures = [source for source in sources if "failure" in source.stem]
    successes = [source for source in sources if "success" in source.stem]
    if len(failures) != 1 or len(successes) != 1:
        parser.error("Input filenames must identify one failure and one success")
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 11.2))
        for row, source in enumerate([failures[0], successes[0]]):
            plot_trace(source, args.output_dir, axes=axes[row], show_legend=row == 1)
        fig.tight_layout(w_pad=2.5, h_pad=2.0)
        save_figure(fig, args.output_dir, "failure_success_evidence")


if __name__ == "__main__":
    main()
