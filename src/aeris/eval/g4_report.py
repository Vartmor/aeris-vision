"""Produce local G4 plots and a report from saved, verified test outputs."""

import html
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
from matplotlib.figure import Figure
import numpy as np

from aeris.calibration.frozen import sha256
from aeris.config import load_config
from aeris.data.manifest import read_manifest
from aeris.eval.gallery import LABELS, TITLES, choose_examples, write_example

_PARTITIONS = {
    "test": ("ID · Hurricane Matthew", "#752d4e", "-", "o"),
    "ood_test": ("OOD · Hurricane Michael", "#356a62", "--", "s"),
}
_VIEWS = {
    "oracle": "Gerçek bina sınırları",
    "candidate": "Modelin bulduğu bina adayları",
}
_ECHEADER = ("Bölüm", "Etiketli örnek", "Ham ECE", "Kalibre ECE", "Ham NLL", "Kalibre NLL")


def _number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    return f"{value:.{digits}f}".replace(".", ",")


def _percent(value: float | None) -> str:
    return f"%{_number(100 * value, 1)}" if value is not None else "—"


def _risk_plot(curves: dict, final: dict, view: str, path: Path) -> None:
    fig = Figure(figsize=(8.2, 4.8), layout="constrained", facecolor="#fbf5f1")
    ax = fig.subplots()
    ax.set_facecolor("#fbf5f1")
    for partition, (label, color, line, marker) in _PARTITIONS.items():
        points = curves[partition][view]
        xs = np.array([point["coverage"] * 100 for point in points])
        ys = np.array([point["risk"] * 100 if point["risk"] is not None else np.nan for point in points])
        ax.step(xs, ys, where="post", color=color, linestyle=line, linewidth=1.8, label=label)
        fixed = final["partitions"][partition]["review"][view]
        if fixed["risk"] is not None:
            ax.scatter(100 * fixed["coverage"], 100 * fixed["risk"], s=88, marker=marker,
                       color=color, edgecolor="#fbf5f1", linewidth=1.2, zorder=5)
    threshold = _number(final["thresholds"][view], 4)
    fig.suptitle(_VIEWS[view], x=0.05, ha="left", fontsize=17, fontweight="bold", color="#3a2030")
    ax.set_title(f"İşaretler, ID kalibrasyonunda seçilen sabit güven eşiğini gösteriyor (τ = {threshold})",
                 loc="left", fontsize=10, color="#624958", pad=13)
    ax.set(xlim=(0, 100), ylim=(0, 101), xlabel="Kapsama (%)", ylabel="Kabul edilenlerde hata oranı (%)")
    ax.grid(alpha=0.35)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="#fbf5f1")
    fig.savefig(path, dpi=150)


def _table(headers: tuple, rows: list[tuple]) -> str:
    head = "".join(f"<th scope=\"col\">{html.escape(str(value))}</th>" for value in headers)
    body = []
    for row in rows:
        label, *cells = row
        data = "".join(f"<td>{html.escape(str(value))}</td>" for value in cells)
        body.append(f"<tr><th scope=\"row\">{html.escape(str(label))}</th>{data}</tr>")
    return "<div class=\"table-wrap\"><table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def _tables(final: dict) -> str:
    reliability, operating, buildings, severe = [], [], [], []
    for partition, (label, _, _, _) in _PARTITIONS.items():
        result = final["partitions"][partition]
        raw, calibrated = (result["oracle_calibration"][key] for key in ("raw", "calibrated"))
        reliability.append((label, _number(raw["samples"]), *(_number(row[key], 4)
                            for key in ("ece", "nll") for row in (raw, calibrated))))
        for view in _VIEWS:
            point = result["review"][view]
            operating.append((
                f"{label} · {_VIEWS[view]}", _number(point["accepted"]), _number(point["review"]),
                _percent(point["coverage"]), _percent(point["risk"]),
                _percent(point["full_coverage_risk"]), _number(point["accepted_unscored"]),
            ))
        row = result["buildings"]
        buildings.append((
            label, _number(row["references"]), _number(row["matches"]), _number(row["unmatched_references"]),
            _percent(result["unmatched_building_rate"]), _number(row["unmatched_candidates"]),
            _number(row["merged_candidates"]), _number(row["merged_references"]),
        ))
        errors = row["severe_damage"]
        severe.append((label, *(_number(errors[key]) for key in (
            "total", "not_detected", "in_merged_candidate", "no_damage", "minor_damage", "major_damage", "destroyed",
        ))))
    return "".join((
        "<h2>Kalibrasyon</h2>",
        _table(_ECHEADER, reliability),
        "<h2>Sabit eşikte kabul ve inceleme</h2>",
        _table(("Bölüm ve sınır kaynağı", "Kabul", "İnceleme", "Kapsama", "Kabul riski",
                "Tam kapsama riski", "Kabul edilen sınıfsız eşleşme"), operating),
        "<h2>Bina eşleştirme</h2>",
        _table(("Bölüm", "Etiketli", "Eşleşen", "Eşleşmeyen bina", "Eşleşmeyen oranı", "Eşleşmeyen aday",
                "Birleşik aday", "Birleşik aday kapsamındaki bina"), buildings),
        "<h2>Ağır hasarlı ve yıkılmış binalar</h2>",
        _table(("Bölüm", "Toplam", "Bulunamadı", "Birleşik adayda", "Hasarsız tahmini", "Hafif hasar tahmini",
                "Ağır hasar tahmini", "Yıkılmış tahmini"), severe),
    ))


def _gallery_cards(examples: dict) -> str:
    cards = []
    for category, title in TITLES.items():
        record = examples.get(category)
        safe_title = html.escape(title)
        if record is None:
            cards.append(f"<article><h3>{safe_title}</h3><p>Bu tanıma uyan örnek bulunamadı.</p></article>")
            continue
        context = [_PARTITIONS[record["partition"]][0], record["sample_id"]]
        context.append("Gerçek etiket: " + ", ".join(
            LABELS[item["damage_class"]] for item in record["references"]
        ))
        if record["predicted_class"] is not None:
            context.append("Tahmin: " + LABELS[record["predicted_class"]])
        if record["confidence"] is not None:
            context.append("Güven " + _percent(record["confidence"]))
        if record["iou"] is not None:
            context.append("IoU " + _number(record["iou"], 2))
        caption = html.escape(" · ".join(context))
        cards.append(
            f"<article><h3>{safe_title}</h3><img src=\"gallery/{record['image']}\" width=\"890\" height=\"360\" "
            f"alt=\"{safe_title}. Solda afet öncesi, sağda afet sonrası görüntü.\" loading=\"lazy\"><p>{caption}</p></article>"
        )
    return "".join(cards)


def _write_html(stage: Path, final: dict, examples: dict) -> None:
    title = "AERIS VISION · G4 değerlendirmesi"
    groups = final["bootstrap"]["groups"]
    text = f"""<!doctype html>
<html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
* {{ box-sizing:border-box }} body {{ margin:0; color:#3a2030; background:#fbf5f1; font:16px/1.58 system-ui,sans-serif }}
main {{ width:min(1120px,calc(100% - 36px)); margin:38px auto 76px }}
h1,h2,h3,p {{ margin:0 }} h1 {{ line-height:1.18 }} h2 {{ margin:30px 0 9px }}
p {{ max-width:80ch; margin:10px 0 }} img {{ width:100%; height:auto; display:block }}
figure {{ margin:16px 0; border:1px solid #dfd0d3 }} .table-wrap {{ overflow-x:auto }}
table {{ width:100%; border-collapse:collapse }} th,td {{ padding:8px; text-align:left; border-bottom:1px solid #dfd0d3 }}
thead {{ background:#f1e9e4 }} tbody th {{ font-weight:600 }} td {{ white-space:nowrap }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,410px),1fr)); gap:18px }}
article {{ padding:12px; border:1px solid #dfd0d3 }} article img {{ margin-top:8px }}
</style>
<main>
<h1>{title}</h1>
<p>Sıcaklık {_number(final["temperature"], 6)} ve iki kabul eşiği, ID kalibrasyon verisinde belirlendi. ID testinde {groups["test"]}, OOD testinde {groups["ood_test"]} bağımsız grup var. Bu gruplardan bootstrap güven aralığı çıkarmadım.</p>
<figure><img src="oracle-risk-coverage.png" alt="Gerçek bina sınırlarında ID ve OOD hata oranı ve kapsama eğrileri." width="1230" height="720"></figure>
<figure><img src="candidate-risk-coverage.png" alt="Modelin bulduğu adaylarda ID ve OOD hata oranı ve kapsama eğrileri." width="1230" height="720"></figure>
{_tables(final)}
<p>Aday kapsamasının paydasında bütün adaylar var. Eşleşmeyen adaylar riskte hata sayılıyor. Sınıfsız binayla eşleşen adaylar hasar riski hesabından çıkarıldı. Kaçırılan gerçek binalar ayrı ölçüldü.</p>
<h2>Yerel hata galerisi</h2>
<p>Her çiftin solunda afet öncesi, sağında afet sonrası görüntü var. Yeşil kutular etiketli binaların, turuncu kutular tahmin edilen adayların sınır kutuları. Kutuların ikisi aynı yerdeyse üst üste gelebilir. Eşik örnekleri gerçek bina sınırlarıyla seçildi. xBD görüntüleri içeren bu galeri yerel kullanım içindir.</p>
<div class="gallery">{_gallery_cards(examples)}</div>
</main></html>
"""
    (stage / "index.html").write_text(text, encoding="utf-8")


def write_report(run: Path, data_config_path: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Report directory already exists: {output}")
    matplotlib.use("Agg")
    final = json.loads((run / "final.json").read_text(encoding="utf-8"))
    config = load_config(data_config_path)["dataset"]
    manifest = Path(config["manifest_path"])
    if sha256(manifest) != final["freeze"]["manifest_sha256"]:
        raise ValueError("Report manifest differs from the frozen evaluation")
    rows = {row["sample_id"]: row for row in read_manifest(manifest)}
    tiles, curves = {}, {}
    for partition in _PARTITIONS:
        paths = run / partition
        result = final["partitions"][partition]
        scores_file, curve_file = paths / "buildings.jsonl", paths / "risk_coverage.json"
        if (
            sha256(scores_file) != result["raw_logits_sha256"]
            or sha256(curve_file) != result["risk_coverage_sha256"]
        ):
            raise ValueError(f"{partition} scores or curves differ from the saved test result")
        tiles[partition] = [json.loads(line) for line in scores_file.read_text(encoding="utf-8").splitlines()]
        curves[partition] = json.loads(curve_file.read_text(encoding="utf-8"))
    examples = choose_examples(tiles, final["temperature"], final["thresholds"]["oracle"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".g4-report-") as working:
        stage = Path(working)
        gallery = stage / "gallery"
        gallery.mkdir()
        for view in _VIEWS:
            _risk_plot(curves, final, view, stage / f"{view}-risk-coverage.png")
        metadata = {
            category: write_example(example, Path(config["raw_dir"]), rows, gallery / f"{category}.png")
            for category, example in examples.items()
        }
        (stage / "gallery.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        _write_html(stage, final, metadata)
        stage.rename(output)
