import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle
import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry
import csv
import paths

POLLING_INTERVAL = 3
API_URL = "https://rest.uniprot.org"

retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise

def submit_id_mapping(from_db, to_db, ids):
    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
    )
    check_response(request)
    return request.json()["jobId"]

def get_next_link(headers):
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

def check_id_mapping_results_ready(job_id):
    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")
        check_response(request)
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] in ("NEW", "RUNNING"):
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(j["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])

def get_id_mapping_results_link(job_id):
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = session.get(url)
    check_response(request)
    return request.json()["redirectURL"]

def decode_results(response, file_format, compressed):
    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]
        elif file_format == "xlsx":
            return [decompressed]
        elif file_format == "xml":
            return [decompressed.decode("utf-8")]
        else:
            return decompressed.decode("utf-8")
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    elif file_format == "xml":
        return [response.text]
    return response.text

def print_progress_batches(batch_index, size, total):
    n_fetched = min((batch_index + 1) * size, total)
    print(f"Fetched: {n_fetched} / {total}")

def get_batch(batch_response, file_format, compressed):
    batch_url = get_next_link(batch_response.headers)
    while batch_url:
        batch_response = session.get(batch_url)
        batch_response.raise_for_status()
        yield decode_results(batch_response, file_format, compressed)
        batch_url = get_next_link(batch_response.headers)

def combine_batches(all_results, batch_results, file_format):
    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results

def get_id_mapping_results_search(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    if "size" in query:
        size = int(query["size"][0])
    else:
        size = 500
        query["size"] = size
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    parsed = parsed._replace(query=urlencode(query, doseq=True))
    url = parsed.geturl()
    request = session.get(url)
    check_response(request)
    results = decode_results(request, file_format, compressed)
    total = int(request.headers["x-total-results"])
    print_progress_batches(0, size, total)
    for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
        results = combine_batches(results, batch, file_format)
        print_progress_batches(i, size, total)
    return results

def get_organism_name(to_entry):
    if 'organism' not in to_entry.keys():
        return 'Unknown'
    organism_dict = to_entry['organism']
    if 'scientificName' in organism_dict.keys() and organism_dict['scientificName']:
        return organism_dict['scientificName']
    elif 'commonName' in organism_dict.keys() and organism_dict['commonName']:
        return organism_dict['commonName']
    else:
        return 'Unknown'

def get_protein_name(to_entry):
    if 'proteinDescription' not in to_entry.keys():
        return 'Unknown'
    protein_name_dict = to_entry['proteinDescription']
    if 'recommendedName' in protein_name_dict.keys() and protein_name_dict['recommendedName']:
        return protein_name_dict['recommendedName']['fullName']['value']
    elif 'alternativeName' in protein_name_dict.keys() and protein_name_dict['alternativeName']:
        return protein_name_dict['alternativeName']['fullName']['value']
    else:
        return 'Unknown'

def get_primary_accession(to_entry):
    if 'primaryAccession' not in to_entry.keys():
        return 'Unknown'
    return to_entry['primaryAccession']

def extract_relevant_info(results):
    data = []
    for result in results.get('results', []):
        from_id = result['from']
        to_entry = result['to']
        primary_accession = get_primary_accession(to_entry)
        organism = get_organism_name(to_entry)
        protein_name = get_protein_name(to_entry)
        data.append([from_id,primary_accession, organism, protein_name])
    return data

def write_to_csv(data, filename=os.path.join(paths.patch_to_score_results_path,'uniprot_data.csv')):
    headers = ['UniProt ID','primary_accession', 'Organism', 'Protein Name']
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"Data written to {filename}")

def split_list(lst, n):
    """Splits a list into smaller chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    # Replace this list with your own UniProt IDs
    uniprot_ids = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path,'uniprots.pkl'))
    all_data = []
    batches = list(split_list(uniprot_ids, 100000))
    for batch in batches:
        # Run the ID mapping and get results
        job_id = submit_id_mapping(
            from_db="UniProtKB_AC-ID", to_db="UniProtKB", ids=batch)
        if check_id_mapping_results_ready(job_id):
            link = get_id_mapping_results_link(job_id)
            results = get_id_mapping_results_search(link)

            # Extract relevant information
            data = extract_relevant_info(results)
            all_data.extend(data)
    # Write data to a CSV file
    write_to_csv(all_data)
