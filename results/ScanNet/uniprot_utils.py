import requests
from xml.etree.ElementTree import fromstring
import csv

# pdb_mapping_url = 'http://www.rcsb.org/pdb/rest/das/pdb_uniprot_mapping/alignment'
#
#
# def get_uniprot_accession_id(response_xml):
#     root = fromstring(response_xml)
#     return next(
#         el for el in root.getchildren()[0].getchildren()
#         if el.attrib['dbSource'] == 'UniProt'
#     ).attrib['dbAccessionId']

# def get_chain_organism(pdb, chain):
#     pdb_mapping_response = requests.get(
#         pdb_mapping_url, params={'query': '%s.%s' % (pdb, chain)}
#     ).text
#     print(pdb_mapping_response)
#     uniprot_id = get_uniprot_accession_id(pdb_mapping_response)
#     name, ecnumber, organism, lineage = get_uniprot_organism(uniprot_id)
#     return (uniprot_id, name, ecnumber, organism, lineage)


uniprot_url = 'http://www.uniprot.org/uniprot/{}.xml'
pdb_mapping_url ='https://1d-coordinates.rcsb.org/graphql?query=%7Balignment(from:PDB_INSTANCE,to:UNIPROT,queryId:%22{pdb}.{chain}%22)%7Btarget_alignment%7Btarget_id%7D%7D%7D'

def get_uniprot_id(pdb,chain):
    url = pdb_mapping_url.format(pdb=pdb.upper(),chain=chain)
    r = requests.get(url)
    uniprotid = r.json()['data']['alignment']['target_alignment'][0]['target_id']
    return uniprotid


def get_uniprot_organism(uniprot_id):
    uinprot_response = requests.get(
        uniprot_url.format(uniprot_id)
    ).text

    response = fromstring(uinprot_response)

    try:
        name = response.find(
            './/{http://uniprot.org/uniprot}recommendedName/{http://uniprot.org/uniprot}fullName'
        ).text
    except:
        name = ''

    try:
        ecnumber = response.find(
            './/{http://uniprot.org/uniprot}recommendedName/{http://uniprot.org/uniprot}ecNumber'
        ).text
    except:
        ecnumber = ''

    try:
        organism = response.find(
            './/{http://uniprot.org/uniprot}organism/{http://uniprot.org/uniprot}name'
        ).text
    except:
        organism = ''

    try:
        lineage = [x.text for x in response.findall(
            './/{http://uniprot.org/uniprot}organism/{http://uniprot.org/uniprot}lineage/'
        )]
    except:
        lineage = ''
    return name, ecnumber, organism, lineage


def get_chain_organism(uniprot_id):
    name, ecnumber, organism, lineage = get_uniprot_organism(uniprot_id)
    return (uniprot_id, name, ecnumber, organism, lineage)

def map_pdb_chains_to_uniprot(input_csv, output_csv="pdb_to_uniprot.csv"):
    """
    Reads input CSV with PDB_ID,CHAIN_ID columns and writes output CSV with UniProt accession.

    Args:
        input_csv (str): Path to input CSV with columns [PDB_ID, CHAIN_ID]
        output_csv (str): Path to save results [PDB_ID, CHAIN_ID, uniprot]
    """
    base_url = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{}"
    rows_out = []

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["PDB_ID"].strip()
            chain = row["CHAIN_ID"].strip()
            uniprot_acc = ""

            try:
                url = base_url.format(pdb_id.lower())
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json().get(pdb_id.lower(), {}).get("UniProt", {})
                    for acc, info in data.items():
                        for mapping in info.get("mappings", []):
                            if mapping.get("chain_id") == chain:
                                uniprot_acc = acc
                                break
                        if uniprot_acc:
                            break
                else:
                    print(f"Failed for {pdb_id}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"Error with {pdb_id} chain {chain}: {e}")

            rows_out.append([pdb_id, chain, uniprot_acc])

    # Write results
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PDB_ID", "CHAIN_ID", "uniprot"])
        writer.writerows(rows_out)

    print(f"Saved {len(rows_out)} rows to {output_csv}")


if __name__ == '__main__':
    uniprot_id, name, ecnumber, organism, lineage = get_chain_organism('1a3x', 'A')