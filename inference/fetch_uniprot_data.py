# import csv
# import time
# from typing import List, Dict, Any, Optional
# import requests
# from tqdm import tqdm


# def uniprot_to_csv(
#     uniprot_ids: List[str],
#     csv_path: str,
#     timeout: int = 60,
#     batch_size: int = 5000,     # ID Mapping supports large batches; 5k is a good chunk
#     max_retries: int = 5,
#     backoff_base: float = 0.8,
#     poll_interval: float = 1.0, # seconds between status polls
# ):
#     """
#     Bulk UniProt fetch using the official ID Mapping API.
#     Saves CSV with: uniprot, primary_accession, Organism, Protein Name
#     """

#     # --- Helpers -------------------------------------------------------------
#     def _clean_ids(ids: List[str]) -> List[str]:
#         out, seen = [], set()
#         for x in ids:
#             if not x:
#                 continue
#             xid = x.strip()
#             if not xid:
#                 continue
#             if xid not in seen:
#                 seen.add(xid)
#                 out.append(xid)
#         return out

#     def _chunks(seq, n):
#         for i in range(0, len(seq), n):
#             yield seq[i:i+n]

#     def _request_with_retries(method: str, url: str, session: requests.Session, **kwargs):
#         delay = backoff_base
#         for attempt in range(max_retries + 1):
#             try:
#                 resp = session.request(method, url, timeout=timeout, **kwargs)
#                 if resp.status_code < 400:
#                     return resp
#                 # Retryable?
#                 if resp.status_code in (429, 500, 502, 503, 504):
#                     time.sleep(delay)
#                     delay *= 2.0
#                     continue
#                 # Non-retryable
#                 return resp
#             except requests.RequestException:
#                 if attempt < max_retries:
#                     time.sleep(delay)
#                     delay *= 2.0
#                 else:
#                     raise

#     def _submit_id_mapping(session: requests.Session, ids: List[str]) -> Optional[str]:
#         """
#         Submit a mapping job: from UniProtKB_AC-ID to UniProtKB entries.
#         Returns jobId.
#         """
#         url = "https://rest.uniprot.org/idmapping/run"
#         data = {
#             "from": "UniProtKB_AC-ID",   # accessions/IDs (e.g., P12345, Q9N...); also handles entry names
#             "to": "UniProtKB",
#             "ids": ",".join(ids),
#         }
#         resp = _request_with_retries("POST", url, session, data=data)
#         if resp.status_code >= 400:
#             return None
#         return resp.json().get("jobId")

#     def _poll_status(session: requests.Session, job_id: str) -> bool:
#         """
#         Poll job status until 'FINISHED' (True) or error (False).
#         """
#         status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
#         while True:
#             resp = _request_with_retries("GET", status_url, session)
#             if resp.status_code >= 400:
#                 return False
#             js = resp.json()
#             if js.get("jobStatus") == "RUNNING":
#                 time.sleep(poll_interval)
#                 continue
#             if js.get("jobStatus") == "FINISHED" or js.get("status") == "FINISHED":
#                 return True
#             # Any other terminal state counts as failure
#             return False

#     def _stream_results(session: requests.Session, job_id: str):
#         """
#         Stream mapped UniProtKB records as JSON (generator of result items).
#         We ask for the rich JSON payload so we can extract recommendedName hierarchy.
#         """
#         # 'stream' delivers all results; add size param if needed to page
#         url = f"https://rest.uniprot.org/idmapping/stream/{job_id}"
#         # Request the full JSON entries; we won't use 'fields=' to avoid 400s on JSON mode
#         params = {"format": "json"}
#         resp = _request_with_retries("GET", url, session, params=params, stream=False)
#         if resp.status_code >= 400:
#             return []
#         js = resp.json()
#         # Responses can be {"results":[{from, to:{...entry...}}, ...]}
#         return js.get("results", [])

#     def _extract_row(entry: Dict[str, Any]) -> Dict[str, str]:
#         """
#         Parse one UniProt entry JSON (the 'to' object) into required fields.
#         """
#         primary_accession = entry.get("primaryAccession", "")

#         organism = ""
#         if "organism" in entry:
#             organism = entry["organism"].get("scientificName", "") or ""

#         protein_name = ""
#         pd = entry.get("proteinDescription") or {}
#         rec = (pd.get("recommendedName") or {}).get("fullName") or {}
#         protein_name = rec.get("value", "") if isinstance(rec, dict) else ""
#         if not protein_name:
#             submitted = pd.get("submittedName") or []
#             if submitted:
#                 protein_name = (submitted[0].get("fullName") or {}).get("value", "") or protein_name
#         if not protein_name:
#             alts = pd.get("alternativeNames") or []
#             if alts:
#                 protein_name = (alts[0].get("fullName") or {}).get("value", "") or protein_name

#         return {
#             "primary_accession": primary_accession,
#             "Organism": organism,
#             "Protein Name": protein_name,
#         }

#     # --- Main ---------------------------------------------------------------
#     cleaned_ids = _clean_ids(uniprot_ids)

#     # Will fill this mapping by accession (resolved) and by original ID for convenience
#     acc_to_row: Dict[str, Dict[str, str]] = {}
#     orig_to_acc: Dict[str, str] = {}  # from original 'from' ID to primary accession

#     session = requests.Session()

#     # Submit in big batches
#     for batch in tqdm(list(_chunks(cleaned_ids, batch_size)), desc="Submitting ID Mapping batches"):
#         job_id = _submit_id_mapping(session, batch)
#         if not job_id:
#             # Mark every ID as error for this batch
#             for uid in batch:
#                 acc_to_row.setdefault(uid, {
#                     "primary_accession": "",
#                     "Organism": "",
#                     "Protein Name": "Error (failed to submit ID mapping)"
#                 })
#             continue

#         # Wait until finished
#         ok = _poll_status(session, job_id)
#         if not ok:
#             for uid in batch:
#                 acc_to_row.setdefault(uid, {
#                     "primary_accession": "",
#                     "Organism": "",
#                     "Protein Name": "Error (mapping job failed)"
#                 })
#             continue

#         # Stream results (mapped)
#         results = _stream_results(session, job_id)
#         for item in results:
#             frm = item.get("from", "")
#             to = item.get("to") or {}
#             row = _extract_row(to)
#             acc = row.get("primary_accession", "")
#             if acc:
#                 acc_to_row[acc] = row
#             if frm and acc:
#                 orig_to_acc[frm] = acc

#     # Write output preserving the original input order (including duplicates)
#     fieldnames = ["uniprot", "primary_accession", "Organism", "Protein Name"]
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for uid in uniprot_ids:
#             # Prefer mapped accession for this original uid
#             acc = orig_to_acc.get(uid) or orig_to_acc.get(uid.strip()) or uid
#             row = acc_to_row.get(acc) or acc_to_row.get(uid) or {}
#             w.writerow({
#                 "uniprot": uid,
#                 "primary_accession": row.get("primary_accession", ""),
#                 "Organism": row.get("Organism", ""),
#                 "Protein Name": row.get("Protein Name", "") or ("Not found" if row == {} else ""),
#             })

#     print(f'wrote UniProt data for {len(cleaned_ids)} unique IDs to {csv_path}')
#     return csv_path

import requests
import csv
from tqdm import tqdm

def uniprot_to_csv(uniprot_ids, csv_path, batch_size=100):
    """
    Given a list of UniProt IDs, fetch:
      - uniprot (the queried ID)
      - primary_accession
      - Organism (scientific name)
      - Protein Name (recommended name when available)

    Save results to a CSV file.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    all_rows = []

    # Break into batches, since UniProt search supports multiple IDs joined by OR
    for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="Fetching UniProt data"):
        batch = uniprot_ids[i:i+batch_size]
        query = " OR ".join(batch)

        params = {
            "query": query,
            "format": "tsv",
            "fields": "organism_name,protein_name",
            "size": min(batch_size,len(batch))  # max size per request
        }
        r = requests.get(base_url, params=params, timeout=60)
        r.raise_for_status()

        lines = r.text.splitlines()
        header, data_lines = lines[0], lines[1:]
        cnt = 0
        for line in data_lines:
            organism, protein_name = line.split("\t")
            all_rows.append({
                "uniprot": batch[cnt],
                "Organism": organism,
                "Protein Name": protein_name
            })
            cnt += 1
            

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uniprot","Organism","Protein Name"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} entries to {csv_path}")
