import requests
from xml.etree.ElementTree import fromstring

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


def get_chain_organism(pdb, chain):
    uniprot_id = get_uniprot_id(pdb,chain)
    name, ecnumber, organism, lineage = get_uniprot_organism(uniprot_id)
    return (uniprot_id, name, ecnumber, organism, lineage)


if __name__ == '__main__':
    uniprot_id, name, ecnumber, organism, lineage = get_chain_organism('1a3x', 'A')