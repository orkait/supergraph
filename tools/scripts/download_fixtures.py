#!/usr/bin/env python3
"""Download test fixtures for supergraph mock testing.

Run once:  python tools/scripts/download_fixtures.py
Total: ~100MB, gitignored under tests/fixtures/
"""

import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

# Relaxed SSL for old certs on some mirrors
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def dl(url, dest, label=""):
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 0:
        return
    tag = label or p.name
    print(f"  {tag}", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "supergraph-fixtures/1.0"})
        with urllib.request.urlopen(req, context=CTX) as r, open(p, "wb") as f:
            f.write(r.read())
        kb = p.stat().st_size // 1024
        print(f"  ({kb}KB)")
    except Exception as e:
        print(f"  FAILED: {e}")
        p.unlink(missing_ok=True)


# ─── Text: Gutenberg ──────────────────────────────────────────────────────────
def download_text():
    print("\n[text] Gutenberg books")
    books = [
        # Fiction
        (1342, "pride-and-prejudice.txt"),
        (11,   "alice-in-wonderland.txt"),
        (84,   "frankenstein.txt"),
        (1661, "sherlock-holmes.txt"),
        (345,  "dracula.txt"),
        (2701, "moby-dick.txt"),
        (98,   "tale-of-two-cities.txt"),
        (174,  "dorian-gray.txt"),
        (1260, "jane-eyre.txt"),
        (1400, "great-expectations.txt"),
        # Non-fiction
        (1232, "the-prince.txt"),
        (3207, "leviathan.txt"),
        (2488, "origin-of-species.txt"),
        (521,  "flatland.txt"),
        (2009, "art-of-war.txt"),
        (1404, "meditations-aurelius.txt"),
        # Short stories
        (1064, "yellow-wallpaper.txt"),
        (2148, "bartleby.txt"),
        (1268, "metamorphosis.txt"),
        (219,  "heart-of-darkness.txt"),
    ]
    d = FIXTURES / "text"
    for eid, name in books:
        dl(f"https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt", d / name)


# ─── PDFs: arXiv (lightweight, <2MB each) ─────────────────────────────────────
def download_pdfs():
    print("\n[pdf] arXiv papers (6 short papers)")
    papers = [
        ("1706.03762", "attention-is-all-you-need.pdf"),
        ("1810.04805", "bert.pdf"),
        ("2005.11401", "rag.pdf"),
        ("2302.13971", "toolformer.pdf"),
        ("1607.00653", "node2vec.pdf"),
        ("1301.3666",  "word2vec.pdf"),
    ]
    d = FIXTURES / "pdf"
    for arxiv_id, name in papers:
        dl(f"https://arxiv.org/pdf/{arxiv_id}", d / name)


# ─── HTML: Wikipedia (mobile = lighter) ───────────────────────────────────────
def download_html():
    print("\n[html] Wikipedia articles")
    articles = [
        ("Transformer_(deep_learning_architecture)", "transformer.html"),
        ("Large_language_model",                     "llm.html"),
        ("Knowledge_graph",                          "knowledge-graph.html"),
        ("Retrieval-augmented_generation",           "rag.html"),
        ("Vector_database",                          "vector-db.html"),
    ]
    d = FIXTURES / "html"
    for slug, name in articles:
        dl(f"https://en.m.wikipedia.org/wiki/{slug}", d / name)


# ─── Markdown: vector DB READMEs ──────────────────────────────────────────────
def download_markdown():
    print("\n[markdown] GitHub READMEs")
    repos = [
        ("facebookresearch/faiss", "main",   "faiss.md"),
        ("chroma-core/chroma",     "main",   "chroma.md"),
        ("qdrant/qdrant",          "master", "qdrant.md"),
        ("milvus-io/milvus",       "master", "milvus.md"),
        ("pgvector/pgvector",      "master", "pgvector.md"),
    ]
    d = FIXTURES / "markdown"
    for repo, branch, name in repos:
        dl(f"https://raw.githubusercontent.com/{repo}/{branch}/README.md", d / name)


# ─── CSV: tabular datasets ────────────────────────────────────────────────────
def download_csv():
    print("\n[csv] Tabular datasets")
    files = [
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
         "iris.csv"),
        ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
         "tips.csv"),
        ("https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
         "countries.csv"),
        ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
         "penguins.csv"),
        ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
         "mpg.csv"),
    ]
    d = FIXTURES / "csv"
    for url, name in files:
        dl(url, d / name)


# ─── Images: Wikimedia Commons CC0 (diverse categories, 320px) ────────────────
def download_images():
    print("\n[images] Wikimedia Commons (diverse, 320px, CC0)")
    d = FIXTURES / "images"
    d.mkdir(parents=True, exist_ok=True)

    if (d / "manifest.json").exists():
        print("  skip (manifest exists)")
        return

    import time as _time

    # Curated: (filename on Commons, label, local name)
    # Using Special:FilePath which is more reliable than /thumb/ URLs
    WIKI = "https://commons.wikimedia.org/wiki/Special:FilePath"
    imgs = [
        # Animals
        ("YellowLabradorLooking_new.jpg",            "dog",       "dog_01.jpg"),
        ("Cat_November_2010-1a.jpg",                 "cat",       "cat_01.jpg"),
        ("Camponotus_flavomarginatus_ant.jpg",       "ant",       "ant_01.jpg"),
        ("Ara_ararauna_Luc_Viatour.jpg",             "parrot",    "parrot_01.jpg"),
        ("Elephants_at_Amboseli_national_park_against_Mount_Kilimanjaro.jpg", "elephant", "elephant_01.jpg"),
        # Vehicles
        ("2012_Fiat_500_Lounge_--_02-22-2012.jpg",     "car",       "car_01.jpg"),
        ("Left_side_of_Flying_Pigeon.jpg",           "bicycle",   "bicycle_01.jpg"),
        ("GoldenGateBridge-001.jpg",                  "bridge",    "bridge_01.jpg"),
        # Food
        ("Eq_it-na_pizza-margherita_sep2005_sml.jpg","pizza",     "pizza_01.jpg"),
        ("Red_Apple.jpg",                            "apple",     "apple_01.jpg"),
        ("Bananas.jpg",                              "banana",    "banana_01.jpg"),
        # Architecture / scenes
        ("Leaning_Tower_of_Pisa.jpg",                "tower",     "tower_01.jpg"),
        ("Taipei_101_2009_amk.jpg",                  "skyscraper","skyscraper_01.jpg"),
        # Nature
        ("Sunflower_from_Silesia2.jpg",              "flower",    "flower_01.jpg"),
        ("Pleiades_large.jpg",                       "stars",     "stars_01.jpg"),
        # Objects
        ("Rubiks_cube_by_keqs.jpg",                  "cube",      "cube_01.jpg"),
        ("Fender_Stratocaster.jpg",                  "guitar",    "guitar_01.jpg"),
        # Art / misc
        ("Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg", "painting", "painting_01.jpg"),
        ("Tux.png",                                  "penguin",   "penguin_01.png"),
        ("Tennis_Racket_and_Balls.jpg",              "tennis",    "tennis_01.jpg"),
    ]
    manifest = []
    for wiki_name, label, fname in imgs:
        url = f"{WIKI}/{wiki_name}?width=320"
        dl(url, d / fname, label=f"{fname} [{label}]")
        _time.sleep(1.5)  # respect Wikimedia rate limits
        if (d / fname).exists():
            manifest.append({"file": fname, "label": label})

    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  manifest.json ({len(manifest)} images)")


# ─── Voice: AI4Bharat Svarah ──────────────────────────────────────────────────
def download_voice():
    print("\n[voice] Multi-language real speech (6 Indian + English)")
    print("  NOTE: Temp downloads ~2GB, cleaned up after extracting 5 clips per lang")
    d = FIXTURES / "voice"
    d.mkdir(parents=True, exist_ok=True)

    import tarfile, shutil, zipfile

    manifest_path = d / "manifest.json"
    if manifest_path.exists():
        print("  skip (manifest exists)")
        return

    manifest = []

    def _extract_slr_tar(lang_code, url, label, audio_ext="mp3"):
        """Download OpenSLR tar.gz, extract 5 clips + transcripts, cleanup."""
        print(f"  [{lang_code}] {label}...")
        archive = d / f"_{lang_code}.tar.gz"
        xdir = d / f"_{lang_code}_x"
        dl(url, archive, label)
        if not archive.exists():
            print(f"    FAILED to download")
            return
        xdir.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(xdir, filter="data")
        # Parse transcripts (Kaldi-style "utt_id text")
        transcripts = {}
        for tf in xdir.rglob("text"):
            for line in tf.read_text(encoding="utf-8", errors="replace").strip().splitlines():
                idx = line.find(" ")
                if idx > 0:
                    transcripts[line[:idx].strip()] = line[idx + 1:].strip()
        for i, src in enumerate(sorted(xdir.rglob(f"*.{audio_ext}"))[:5]):
            fname = f"{lang_code}_{i:02d}.{audio_ext}"
            shutil.copy2(src, d / fname)
            kb = (d / fname).stat().st_size // 1024
            print(f"    {fname} ({kb}KB)")
            manifest.append({"file": fname, "lang": lang_code, "transcript": transcripts.get(src.stem, "")})
        archive.unlink(missing_ok=True)
        shutil.rmtree(xdir, ignore_errors=True)

    def _extract_slr_zip(lang_code, url, label):
        """Download OpenSLR zip, extract 5 clips + transcripts, cleanup."""
        print(f"  [{lang_code}] {label}...")
        archive = d / f"_{lang_code}.zip"
        xdir = d / f"_{lang_code}_x"
        dl(url, archive, label)
        if not archive.exists():
            print(f"    FAILED to download")
            return
        xdir.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(xdir)
        # Parse line_index TSV (Google Crowdsourced format: "filename \t text")
        transcripts = {}
        for tf in list(xdir.rglob("line_index*.tsv")) + list(xdir.rglob("line_index*.csv")):
            for line in tf.read_text(encoding="utf-8", errors="replace").strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    transcripts[Path(parts[0]).stem] = parts[1]
        for i, src in enumerate(sorted(xdir.rglob("*.wav"))[:5]):
            fname = f"{lang_code}_{i:02d}.wav"
            shutil.copy2(src, d / fname)
            kb = (d / fname).stat().st_size // 1024
            print(f"    {fname} ({kb}KB)")
            manifest.append({"file": fname, "lang": lang_code, "transcript": transcripts.get(src.stem, "")})
        archive.unlink(missing_ok=True)
        shutil.rmtree(xdir, ignore_errors=True)

    # Hindi: SLR118 eval (62MB, Gram Vaani telephone, real Indian speakers)
    _extract_slr_tar("hi",
        "https://openslr.org/resources/118/GV_Eval_3h.tar.gz",
        "OpenSLR 118 Hindi eval (62MB)")

    # English: SLR83 midlands_english_female (103MB, UK English)
    _extract_slr_zip("en",
        "https://openslr.trmal.net/resources/83/midlands_english_female.zip",
        "OpenSLR 83 English midlands (103MB)")

    # Tamil: SLR65 (603MB, Indian Tamil speakers)
    _extract_slr_zip("ta",
        "https://openslr.trmal.net/resources/65/ta_in_male.zip",
        "OpenSLR 65 Tamil (603MB)")

    # Telugu: SLR66 (505MB, Indian Telugu speakers)
    _extract_slr_zip("te",
        "https://openslr.trmal.net/resources/66/te_in_female.zip",
        "OpenSLR 66 Telugu (505MB)")

    # Marathi: SLR64 (712MB, Indian Marathi speakers)
    _extract_slr_zip("mr",
        "https://openslr.trmal.net/resources/64/mr_in_female.zip",
        "OpenSLR 64 Marathi (712MB)")

    # Gujarati skipped - 895MB archive, requires >2GB free disk for extraction

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  manifest.json ({len(manifest)} clips, 5 languages)")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Downloading fixtures to {FIXTURES}/")
    download_text()
    download_pdfs()
    download_html()
    download_markdown()
    download_csv()
    download_images()
    download_voice()

    # Ensure gitignored
    gi = ROOT / "tests" / "fixtures" / ".gitkeep"
    gi.parent.mkdir(parents=True, exist_ok=True)

    total = sum(f.stat().st_size for f in FIXTURES.rglob("*") if f.is_file()) // (1024 * 1024)
    print(f"\nDone. Total: ~{total}MB")
