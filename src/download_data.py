"""
Best-effort downloader for the Mechanical MNIST -- Uniaxial Extension dataset
(Lejeune Lab, Boston University OpenBU repository, handle 2144/38693).

IMPORTANT: OpenBU (DSpace) blocked a plain automated fetch with HTTP 403
during development of this script, so this is a best-effort helper, not a
guaranteed one-shot solution. If it fails:

    1. Open https://open.bu.edu/handle/2144/38693 in a browser.
    2. Download these three files manually into data/raw/:
         - MNIST_input_files.zip
         - FEA_psi_results.zip
         - FEA_rxnforce_results.zip
    3. Re-run this script with --extract-only to unzip what you downloaded:
         python src/download_data.py --extract-only

Citation (please cite if you use this data):
  E. Lejeune, "Mechanical MNIST: A benchmark dataset for mechanical
  metamodels", Extreme Mechanics Letters, 2020.
  https://doi.org/10.1016/j.eml.2020.100659
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import urllib.error
import zipfile

BASE_URL = "https://open.bu.edu/bitstream/handle/2144/38693"
FILES = [
    "MNIST_input_files.zip",
    "FEA_psi_results.zip",
    "FEA_rxnforce_results.zip",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-download-script/1.0)"}


def download_file(fname: str, out_dir: str) -> bool:
    url = f"{BASE_URL}/{fname}?sequence=1&isAllowed=y"
    dest = os.path.join(out_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {fname} already present")
        return True
    print(f"[fetch] {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        print(f"[ok] saved {dest} ({os.path.getsize(dest)} bytes)")
        return True
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"[FAIL] could not download {fname}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def extract_all(out_dir: str):
    for fname in os.listdir(out_dir):
        if fname.endswith(".zip"):
            path = os.path.join(out_dir, fname)
            try:
                with zipfile.ZipFile(path) as zf:
                    target = os.path.join(out_dir, fname[:-4])
                    zf.extractall(target)
                    print(f"[extract] {fname} -> {target}/")
            except zipfile.BadZipFile:
                print(f"[WARN] {fname} is not a valid zip (partial download?) -- skipping")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--extract-only", action="store_true",
                     help="Skip downloading, just unzip whatever is already in --out")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if not args.extract_only:
        ok = True
        for fname in FILES:
            ok &= download_file(fname, out_dir)
        if not ok:
            print(
                "\nSome files failed to download automatically (this host "
                "returned HTTP 403 to a scripted request during testing). "
                "Please download them by hand from "
                "https://open.bu.edu/handle/2144/38693 into "
                f"{out_dir}/ and re-run with --extract-only.",
                file=sys.stderr,
            )

    extract_all(out_dir)
    print("\nDone. Run `python src/dataset.py --inspect data/raw` to see what "
          "was extracted, then check the parsing assumptions documented at "
          "the top of src/dataset.py (especially _load_reaction_force).")


if __name__ == "__main__":
    main()
