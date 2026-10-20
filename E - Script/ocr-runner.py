#!/usr/bin/env python3
"""
ocr_ingest.py - send PDFs to Azure Document Intelligence (layout model) via the SDK,
store the raw JSON and the service's own markdown (tables included), log every call.

Setup:
    pip install azure-ai-documentintelligence
    export AZURE_DI_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com"
    export AZURE_DI_KEY="<key>"

Usage:
    python ocr_ingest.py corpus/pdfs/                 # every PDF in a folder
    python ocr_ingest.py a.pdf b.pdf --force          # re-parse even if output exists
    python ocr_ingest.py corpus/pdfs/ --timeout 900   # seconds per document

Output per document, in corpus/parsed/<doc_id>/ :
    raw.json      full service response
    document.md   service markdown, with [[PAGE n]] markers in place of page breaks
    pages.jsonl   one record per page (page, text), for quote checking
    meta.json     file hash, model, counts, seconds

Every HTTP call (submit, each poll) and every document outcome is appended to
logs/ocr_calls.jsonl. Skipped, done and failed documents are counted and reported at the end;
exit code is 1 if any failed. The key is never logged.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat

MODEL = "prebuilt-layout"
PAGE_BREAK = "<!-- PageBreak -->"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CallLog:
    """Append-only JSONL log."""

    def __init__(self, path, run_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def __call__(self, **event):
        line = {"ts": utc_now(), "run_id": self.run_id, **event}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def analyze(client, log, doc_id, data, model, timeout_s):
    """Submit one PDF, wait for the result (SDK polls), return the AnalyzeResult."""
    t0 = time.time()

    def on_response(resp):  # runs for the submit call and for every poll
        r = resp.http_response
        log(doc_id=doc_id, event="http", method=resp.http_request.method, http_status=r.status_code,
            request_id=r.headers.get("apim-request-id") or r.headers.get("x-ms-request-id"),
            elapsed_s=round(time.time() - t0, 2))

    poller = client.begin_analyze_document(
        model, AnalyzeDocumentRequest(bytes_source=data),
        output_content_format=DocumentContentFormat.MARKDOWN,
        string_index_type="unicodeCodePoint",
        raw_response_hook=on_response)
    return poller.result(timeout=timeout_s)  # raises TimeoutError if not done in time


def split_pages(markdown):
    """Service markdown -> list of page texts (the service marks page breaks with a comment)."""
    return [p.strip() for p in markdown.split(PAGE_BREAK)]


def write_outputs(out_dir, doc_id, result, pages, meta):
    d = out_dir / doc_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw.json").write_text(json.dumps(result.as_dict(), ensure_ascii=False), encoding="utf-8")
    (d / "document.md").write_text(
        "\n\n".join(f"[[PAGE {i}]]\n{t}" for i, t in enumerate(pages, 1)) + "\n", encoding="utf-8")
    (d / "pages.jsonl").write_text(
        "".join(json.dumps({"page": i, "text": t}, ensure_ascii=False) + "\n"
                for i, t in enumerate(pages, 1)), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="PDF files or folders of PDFs")
    ap.add_argument("--out", default="corpus/parsed")
    ap.add_argument("--log", default="logs/ocr_calls.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds to wait per document")
    ap.add_argument("--force", action="store_true", help="re-parse documents that already have output")
    a = ap.parse_args()

    endpoint, key = os.environ.get("AZURE_DI_ENDPOINT"), os.environ.get("AZURE_DI_KEY")
    if not endpoint or not key:
        sys.exit("Set AZURE_DI_ENDPOINT and AZURE_DI_KEY")

    run_id = "ocr-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = CallLog(a.log, run_id)
    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pdfs = []
    for p in map(Path, a.paths):
        pdfs += sorted(p.glob("*.pdf")) if p.is_dir() else [p]

    done, skipped, failed = [], [], []
    for pdf in pdfs:
        doc_id = pdf.stem
        # skip-if-done: meta.json is written last, so its presence means the document finished
        if (out / doc_id / "meta.json").exists() and not a.force:
            skipped.append(doc_id)
            log(doc_id=doc_id, event="skipped", reason="already parsed (use --force)")
            continue
        try:
            data = pdf.read_bytes()
            t0 = time.time()
            result = analyze(client, log, doc_id, data, a.model, a.timeout)
            pages = split_pages(result.content)
            meta = {"doc_id": doc_id, "source_file": str(pdf),
                    "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                    "run_id": run_id, "model_id": result.model_id, "api_version": result.api_version,
                    "n_pages": len(result.pages or []), "n_md_pages": len(pages),
                    "n_tables": len(result.tables or []), "seconds": round(time.time() - t0, 1)}
            write_outputs(out, doc_id, result, pages, meta)
            done.append(doc_id)
            log(doc_id=doc_id, event="doc_done", **{k: v for k, v in meta.items() if k != "doc_id"})
            print(f"ok   {doc_id}: {meta['n_pages']} pages, {meta['n_tables']} tables")
        except Exception as e:  # keep going; the failure is logged and counted
            failed.append(doc_id)
            log(doc_id=doc_id, event="doc_failed", error=repr(e))
            print(f"FAIL {doc_id}: {e!r}", file=sys.stderr)

    log(event="run_done", done=len(done), skipped=len(skipped), failed=len(failed))
    print(f"\n{len(done)} done, {len(skipped)} skipped, {len(failed)} failed (of {len(pdfs)})")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()