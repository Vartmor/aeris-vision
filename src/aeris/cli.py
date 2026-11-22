import argparse
import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from aeris.config import ConfigError, load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aeris",
        description="AERIS VISION uydu görüntüsü analizi için proje araçları.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('aeris-vision')}"
    )
    commands = parser.add_subparsers(dest="command")
    config = commands.add_parser("config", help="YAML konfigürasyonunu denetle.")
    config_commands = config.add_subparsers(required=True)
    check = config_commands.add_parser("check", help="YAML dosyasını okuyup denetle.")
    check.add_argument("--config", required=True, type=Path, help="YAML dosyası.")
    check.set_defaults(handler=_config_check)

    data = commands.add_parser("data", help="xBD verisini denetle.")
    data_commands = data.add_subparsers(required=True)
    prepare = data_commands.add_parser(
        "prepare", help="Yerel xBD arşivini doğrula ve veri dizinine aç."
    )
    prepare.add_argument("--config", required=True, type=Path, help="Veri konfigürasyonu.")
    prepare.add_argument("--archive", required=True, type=Path, help="Yerel tar veya tar.gz.")
    prepare.add_argument("--sha256", help="Beklenen arşiv SHA-256 değeri.")
    prepare.set_defaults(handler=_data_prepare)

    audit = data_commands.add_parser(
        "audit", help="Görüntü çiftlerini denetle, manifest ve denetim raporu yaz."
    )
    audit.add_argument("--config", required=True, type=Path, help="Veri konfigürasyonu.")
    audit.add_argument(
        "--no-hash", action="store_true", help="Dosyaların SHA-256 değerlerini hesaplama."
    )
    audit.add_argument(
        "--check-targets",
        action="store_true",
        help="Üretilen maskeleri paketteki resmî target PNG'leriyle karşılaştır.",
    )
    audit.set_defaults(handler=_data_audit)

    verify = data_commands.add_parser(
        "verify", help="Kaydedilen veri denetimini güncel ham paketle karşılaştır."
    )
    verify.add_argument("--config", required=True, type=Path, help="Veri konfigürasyonu.")
    verify.set_defaults(handler=_data_verify)

    split = commands.add_parser("split", help="Veri bölmesini oluştur veya doğrula.")
    split_commands = split.add_subparsers(dest="split_command", required=True)
    for name, help_text in (
        ("make", "Manifesti böl ve sonucu dondur."),
        ("verify", "Dondurulmuş bölmeyi güncel manifestle doğrula."),
    ):
        subcommand = split_commands.add_parser(name, help=help_text)
        subcommand.add_argument("--config", required=True, type=Path, help="Veri konfigürasyonu.")
        subcommand.set_defaults(handler=_split)

    train = commands.add_parser("train", help="Bina lokalizasyon modelini eğit.")
    train.add_argument("--task", required=True, choices=("localization",))
    train.add_argument("--config", required=True, type=Path, help="Model konfigürasyonu.")
    train.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    train.add_argument("--run-dir", required=True, type=Path, help="Yeni yerel koşu dizini.")
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--train-tiles", type=int, help="Train bölümünden seçilecek görüntü sayısı.")
    train.add_argument("--thresholds", type=float, nargs="+", default=[0.5],
                       help="Validation üzerinde karşılaştırılacak bina olasılık eşikleri.")
    train.add_argument("--validation-masks", action="store_true", help="Seçilen modelin validation maskelerini yaz.")
    train.add_argument("--cache-dir", type=Path, help="Yerel ve yeniden kullanılabilir RGB/maske önbelleği.")
    train.add_argument("--tiny", action="store_true", help="32 eğitim yamasında overfit testi.")
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser(
        "evaluate", help="xView2 uyumlu piksel skorunu hesapla."
    )
    evaluate.add_argument(
        "--predictions", required=True, type=Path, help="Tahmin PNG dizini."
    )
    evaluate.add_argument("--targets", required=True, type=Path, help="Hedef PNG dizini.")
    evaluate.add_argument("--output", type=Path, help="JSON çıktı dosyası.")
    evaluate.set_defaults(handler=_evaluate)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        return args.handler(args)
    except ConfigError as error:
        print(f"Konfigürasyon hatası — {error}", file=sys.stderr)
        return 2


def _config_check(args: argparse.Namespace) -> int:
    load_config(args.config)
    print(f"Konfigürasyon okunabildi — {args.config}")
    return 0


def _dataset_section(config: Path) -> dict[str, object]:
    dataset = load_config(config).get("dataset")
    if not isinstance(dataset, dict):
        raise ConfigError(f"{config}: 'dataset' eşlemesi bulunamadı.")
    return dataset


def _source_package(dataset: dict[str, object], config: Path) -> str:
    source_package = dataset.get("source_package")
    if not isinstance(source_package, str) or not source_package:
        raise ConfigError(f"{config}: dataset.source_package metin olmalı.")
    return source_package


def _data_prepare(args: argparse.Namespace) -> int:
    from tarfile import TarError

    from aeris.data.archive import ArchiveError, prepare_archive

    dataset = _dataset_section(args.config)
    raw_dir = _config_path(dataset, "raw_dir", args.config)
    source_package = _source_package(dataset, args.config)
    try:
        receipt = prepare_archive(
            args.archive, raw_dir, source_package=source_package, expected_sha256=args.sha256
        )
    except (ArchiveError, OSError, TarError) as error:
        print(f"Arşiv hazırlama hatası — {error}", file=sys.stderr)
        return 2

    print(f"xBD arşivi hazırlandı — {raw_dir}")
    print(f"Arşiv SHA-256 — {receipt['archive_sha256']}")
    return 0


def _data_audit(args: argparse.Namespace) -> int:
    from aeris.data.manifest import INVALID, WARNING, build_manifest, write_manifest

    dataset = _dataset_section(args.config)
    raw_dir = _config_path(dataset, "raw_dir", args.config)
    manifest_path = _config_path(dataset, "manifest_path", args.config)
    audit_path = _config_path(dataset, "audit_path", args.config)
    source_package = _source_package(dataset, args.config)

    if not raw_dir.is_dir():
        print(f"Veri dizini bulunamadı — {raw_dir}", file=sys.stderr)
        return 2
    result = build_manifest(
        raw_dir,
        source_package,
        compute_hashes=not args.no_hash,
        check_targets=args.check_targets,
    )
    if not result.records:
        print(f"Etiket dosyası bulunamadı — {raw_dir}/**/labels/*.json", file=sys.stderr)
        return 2
    write_manifest(result, manifest_path, audit_path)

    summary = result.summary
    statuses = summary["status_counts"]
    print(
        f"Manifest yazıldı — {manifest_path} · {summary['samples']} örnek, "
        f"{statuses[WARNING]} uyarılı, {statuses[INVALID]} geçersiz"
    )
    print(f"Denetim raporu — {audit_path}")
    print()
    print(f"{'Olay':<28} {'Örnek':>6} {'Bina':>7}  no / minor / major / destroyed / unclassified")
    for event, info in summary["events"].items():
        counts = info["damage_label_counts"]
        print(
            f"{event:<28} {info['samples']:>6} {info['pre_buildings']:>7}  "
            f"{counts['no_damage']} / {counts['minor_damage']} / {counts['major_damage']} / "
            f"{counts['destroyed']} / {counts['un_classified']}"
        )
    return 0


def _data_verify(args: argparse.Namespace) -> int:
    from aeris.data.manifest import ManifestVerificationError, verify_manifest

    dataset = _dataset_section(args.config)
    raw_dir = _config_path(dataset, "raw_dir", args.config)
    manifest_path = _config_path(dataset, "manifest_path", args.config)
    audit_path = _config_path(dataset, "audit_path", args.config)
    source_package = _source_package(dataset, args.config)
    try:
        result = verify_manifest(raw_dir, source_package, manifest_path, audit_path)
    except (ManifestVerificationError, OSError) as error:
        print(f"Veri doğrulama hatası — {error}", file=sys.stderr)
        return 2

    print(f"Manifest ve denetim raporu doğrulandı — {result['samples']} örnek")
    print(f"Manifest SHA-256 — {result['manifest_sha256']}")
    for phase, counts in result["target_check"].items():
        print(
            f"{phase}: {counts['samples_compared']} hedef karşılaştırıldı, "
            f"{counts['mismatched_pixels']} farklı piksel"
        )
    return 0


def _split(args: argparse.Namespace) -> int:
    from aeris.splits.planner import SplitError
    from aeris.splits.storage import freeze_split, plan_from_manifest, verify_split

    config = load_config(args.config)
    dataset, split = config.get("dataset"), config.get("split")
    if not isinstance(dataset, dict) or not isinstance(split, dict):
        raise ConfigError(f"{args.config}: 'dataset' ve 'split' eşlemeleri gerekli.")
    manifest_path = _config_path(dataset, "manifest_path", args.config)
    output_path = _config_path(split, "output_path", args.config, section_name="split")
    fractions = split.get("id_fractions")
    if not isinstance(fractions, dict):
        raise ConfigError(f"{args.config}: split.id_fractions eşleme olmalı.")
    ood_event = split.get("ood_event")
    if ood_event is not None and (not isinstance(ood_event, str) or not ood_event.strip()):
        raise ConfigError(f"{args.config}: split.ood_event dolu bir metin ya da null olmalı.")

    id_groups = split.get("id_evaluation_groups")
    if id_groups is not None and not isinstance(id_groups, dict):
        raise ConfigError(f"{args.config}: split.id_evaluation_groups eşleme olmalı.")

    try:
        plan = plan_from_manifest(
            manifest_path, fractions, ood_event=ood_event, id_evaluation_groups=id_groups
        )
        if args.split_command == "make":
            freeze_split(output_path, plan)
        else:
            verify_split(output_path, plan)
    except (OSError, SplitError) as error:
        print(f"Veri bölme hatası — {error}", file=sys.stderr)
        return 2

    action = "donduruldu" if args.split_command == "make" else "doğrulandı"
    print(f"Bölme {action} — {output_path}")
    print(f"OOD olayı — {plan['ood_event']}")
    print(f"Split SHA-256 — {plan['split_sha256']}")
    print()
    for name, stats in plan["summary"].items():
        print(f"{name:<12} {stats['samples']:>6} örnek · {stats['groups']} grup")
    excluded = plan["excluded"]
    invalid = sum(excluded["invalid_manifest_records"].values())
    related = len(excluded["groups_linked_to_ood"])
    print(f"Dağıtım dışı — {invalid} geçersiz kayıt · {related} OOD bağlantılı kayıt")
    return 0


def _train(args: argparse.Namespace) -> int:
    if args.epochs < 1 or args.workers < 0 or not 0 <= args.seed < 2**32:
        raise ConfigError("epochs pozitif, workers sıfır veya pozitif, seed 32 bit işaretsiz olmalı.")
    if args.train_tiles is not None and args.train_tiles < 1:
        raise ConfigError("train-tiles pozitif olmalı.")
    if not all(0 < value < 1 for value in args.thresholds):
        raise ConfigError("Olasılık eşikleri 0 ile 1 arasında olmalı.")
    if args.tiny and (args.train_tiles is not None or args.validation_masks or args.thresholds != [0.5]):
        raise ConfigError("Tiny testi 32 eğitim yaması ve sabit 0,5 eşikle çalışır.")
    try:
        from aeris.train.localization import train_localization
    except ModuleNotFoundError as error:
        if error.name not in {"torch", "torchvision"}:
            raise
        print("Eğitim PyTorch ve torchvision gerektiriyor. Ortam kaydı: docs/environment.md", file=sys.stderr)
        return 2

    try:
        train_localization(args)
    except ConfigError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Eğitim hatası — {error}", file=sys.stderr)
        return 2
    print(f"Eğitim koşusu kaydedildi — {args.run_dir}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from aeris.eval.xview2 import score_directories

    try:
        report = score_directories(args.predictions, args.targets)
        output = json.dumps(report.to_dict(), indent=2) + "\n"
        if args.output is None:
            print(output, end="")
        else:
            args.output.write_text(output, encoding="utf-8")
            print(f"xView2 uyumlu yerel skor yazıldı — {args.output}")
    except (OSError, ValueError) as error:
        print(f"Değerlendirme hatası — {error}", file=sys.stderr)
        return 2
    return 0


def _config_path(
    section: dict[str, object], key: str, config: Path, *, section_name: str = "dataset"
) -> Path:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{config}: {section_name}.{key} metin olmalı.")
    return Path(value)
