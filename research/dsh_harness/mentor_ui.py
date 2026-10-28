"""Run each mentor use case in a new dsh web session (visible Chrome) and save the visible answer.

Usage: mentor_ui.py <dsh-web.log> <usecases.jsonl> <outdir> [first_id]

Run with Playwright from uv, outside the project venv:
    uv run --no-project --with playwright python research/dsh_harness/mentor_ui.py \
        ~/.local/share/hpc-ssh/dsh-web.log data/mentor_usecases/curiosity/usecases.jsonl <outdir>
The script writes <outdir>/runs.json and one screenshot per case. Do not close the Chrome window during the run.
"""
import json
import re
import sys
import time

from playwright.sync_api import sync_playwright

LOG, CASES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FIRST = sys.argv[4] if len(sys.argv) > 4 else None
url = re.findall(r"http://127\.0\.0\.1:3080/\?token=[A-Za-z0-9_-]+", open(LOG).read())[-1]
cases = [json.loads(l) for l in open(CASES) if l.strip()]
if FIRST:
    cases = cases[[c["id"] for c in cases].index(FIRST):]


def prompt(c):
    text = c["user_request"]
    if c.get("context"):
        text += "\n\nContext: " + c["context"]
    return text


def composer(pg):
    for sel in ["textarea", "[contenteditable=true]"]:
        loc = pg.locator(sel)
        if loc.count():
            return loc.last
    raise RuntimeError("no chat input found")


def wait_stable(pg, quiet=15, limit=300):
    """Wait until the page text does not change for `quiet` seconds."""
    last, since, start = None, time.time(), time.time()
    while time.time() - start < limit:
        text = pg.inner_text("body")
        if text != last:
            last, since = text, time.time()
        elif time.time() - since > quiet:
            return text
        time.sleep(1)
    return last


runs = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=False, slow_mo=100)
    pg = b.new_page(viewport={"width": 1300, "height": 850})
    pg.goto(url)
    pg.wait_for_timeout(4000)
    for c in cases:
        print(f"===== {c['id']}: {c['user_request'][:90]}", flush=True)
        pg.get_by_text("New Session", exact=True).first.click()
        pg.wait_for_timeout(2500)
        before = pg.inner_text("body")
        start = time.time()
        box = composer(pg)
        box.click()
        box.fill(prompt(c))
        pg.keyboard.press("Enter")
        after = wait_stable(pg)
        answer = after[len(before):] if after.startswith(before) else after[-4000:]
        runs.append({"id": c["id"], "start": start, "end": time.time(), "visible": answer.strip()})
        pg.screenshot(path=f"{OUT}/{c['id']}.png")
        json.dump(runs, open(f"{OUT}/runs.json", "w"), indent=1)
        print(answer.strip()[:600], flush=True)
    print("===== done", flush=True)
    pg.wait_for_timeout(3000)
    b.close()
