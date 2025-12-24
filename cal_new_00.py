import os
import traceback
import multiprocessing as mp
import subprocess
from Bio.Blast import NCBIXML
import xml.etree.ElementTree as ET
from typing import Optional, List
import pandas as pd
from Bio.PDB import PDBParser
from pymol import cmd
import json
import pickle
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem
import selfies as sf
import re
import numpy as np
import sys
from pubchempy import get_compounds
sys.path.append("/work19/bai/baitokyotech/SaProt/utils")
from foldseek_util import get_struc_seq
foldseek_path = "/work18/baitokyotech/software/anaconda3/envs/interaction/bin/foldseek"


# Load vdW parameter dictionaries
with open("./dic/vdw_params_ligand.json", "r") as f:
    ligand_vdw_params = json.load(f)

def run_plip(pdb_file: str, output_file_name: str, out_dir: str) -> None:
    """Run PLIP with the specified input file and output name."""
    command = ["plip", "-f", pdb_file, "--name", output_file_name, "-o", out_dir, "-x"]
    try:
        subprocess.run(command, check=True)
        print(f"PLIP successfully processed {pdb_file}, output saved as {output_file_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running PLIP: {e}")

def run_get_static_contacts(pdb_file: str, output_file_name: str) -> None:
    """Run get_static_contacts.py for vdW interactions."""
    ligand_resname = extract_ligand_name(pdb_file)
    command = [
        "get_static_contacts.py",
        "--structure", pdb_file,
        "--output", f"{output_file_name}",
        "--sele", "protein",
        "--sele2", f"resname {ligand_resname}",
        "--itypes", "vdw",
        "--distout"
    ]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
        print("get_static_contacts.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running get_static_contacts.py: {e}")

# Helper Functions
def parse_xml(file_path: str) -> ET.Element:
    """Parse the XML file and return the root element."""
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except Exception as e:
        raise ValueError(f"Error parsing XML file: {e}")

def extract_text(element: Optional[ET.Element], default: str = 'N/A') -> str:
    """Safely extract text from an XML element, with a default fallback."""
    return element.text.strip() if element is not None and element.text else default

def extract_ligand_name(pdb_file: str) -> str:
    """Extract the ligand residue name from the PDB file."""
    cmd.load(pdb_file, "structure")
    cmd.select("organic_molecules", "organic")
    model = cmd.get_model("organic_molecules").atom
    resname = model[0].resn
    cmd.delete("all")

    return resname

def get_first_atom_index(structure, ligand_resname: str) -> Optional[int]:
    """Retrieve the first atom index of the ligand from the PDB structure."""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.resname == ligand_resname:
                    return min(atom.serial_number for atom in residue)
    return None


def get_atom_index(pdb_file, resname, atom_name):
    """
    Extracts the atom index for a given residue and atom name in a PDB file.

    Parameters:
        pdb_file (str): Path to the PDB file.
        resname (str): Ligand residue name (e.g., 'GOL').
        atom_name (str): Atom name (e.g., 'O1').

    Returns:
        int: Atom index if found, else None.
    """
    with open(pdb_file, "r") as f:
        for line in f:
            if resname in line and ("ATOM" in line[:6] or "HETATM" in line[:6]):
                extracted_atom_name = line[12:16].strip()
                if extracted_atom_name == atom_name:
                    atom_index = int(line[6:11].strip())  # Extract serial number
                    print(f"✅ Found '{atom_name}' in '{resname}' at index: {atom_index}")
                    return atom_index

    print(f"❌ Error: Atom '{atom_name}' NOT found in Ligand '{resname}'!")
    return None


# Process PLIP Results
def process_interaction(interaction: ET.Element, interaction_type: int, first_atom_index: int) -> List[str]:
    """Process and return details of a single interaction."""
    results = []
    resnr = extract_text(interaction.find('resnr'))  # Protein residue number
    resnr_0_based = int(resnr) - 1 if resnr.isdigit() else 'N/A'
    dist = extract_text(interaction.find('dist'))  # General distance
    ligand_atom_indices = []

    if interaction_type in [6, 5]:  # Hydrophobic interactions or halogen bonds
        lig_atom_index = extract_text(interaction.find('ligcarbonidx' if interaction_type == 6 else 'donoridx'))
        if lig_atom_index.isdigit():
            ligand_atom_indices = [int(lig_atom_index) - first_atom_index]

        if interaction_type == 5:  # Handle halogen bonds specifically
            protisdon = extract_text(interaction.find('protisdon'))
            lig_atom_index = extract_text(interaction.find('donoridx' if protisdon == 'False' else 'acceptoridx'))
            if lig_atom_index.isdigit():
                ligand_atom_indices = [int(lig_atom_index) - first_atom_index]

    elif interaction_type == 1:  # Hydrogen bonds
        dist = extract_text(interaction.find('dist_d-a'))  # Distance between donor and acceptor
        protisdon = extract_text(interaction.find('protisdon'))  # Check if protein is donor
        lig_atom_index = extract_text(interaction.find('donoridx' if protisdon == 'False' else 'acceptoridx'))
        if lig_atom_index.isdigit():
            ligand_atom_indices = [int(lig_atom_index) - first_atom_index]

    elif interaction_type in [2, 3, 4]:  # Salt bridges, pi-stacks, pi-cation interactions
        lig_idx_list = interaction.find('lig_idx_list')
        if lig_idx_list is not None:
            ligand_atom_indices = [
                int(idx.text) - first_atom_index for idx in lig_idx_list.findall('idx') if idx.text.isdigit()
            ]
            ligand_atom_indices = sorted(set(ligand_atom_indices))  # Sort and remove duplicates

        if interaction_type == 3:  # Pi-stacks
            dist = extract_text(interaction.find('centdist'))  # Central distance in pi-stacks

    if  ligand_atom_indices:
        for atom_index in ligand_atom_indices:
            results.append([interaction_type, atom_index, resnr_0_based, dist, 'N/A', 0.000001])

    return results

def process_file_pair(xml_file_name: str, pdb_file: str) -> List[str]:
    """Process PLIP XML and PDB file pair."""
    root = parse_xml(f"{xml_file_name}")
    
    binding_site = root.find(".//bindingsite")
    if binding_site is not None and binding_site.get("has_interactions") == "False":
        print(f"No interactions reported. Returning empty results.")
        return []

    interactions = root.find('.//interactions')
    ligand_resname = extract_text(root.find('.//identifiers/hetid'))

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ligand_structure', pdb_file)
    first_atom_index = get_first_atom_index(structure, ligand_resname)

    interaction_types_map = {
        'hydrogen_bonds': 1,
        'salt_bridges': 2,
        'pi_stacks': 3,
        'pi_cation_interactions': 4,
        'halogen_bonds': 5,
        'hydrophobic_interactions': 6
    }

    results = []
    for interaction_name, interaction_type in interaction_types_map.items():
        interaction_group = interactions.find(interaction_name)
        if interaction_group is not None:
            for interaction in interaction_group:
                results.extend(process_interaction(interaction, interaction_type, first_atom_index))

    return results

def load_tsv_data(tsv_file_name):
    # Check if file exists and is non-empty
    if not os.path.exists(tsv_file_name) or os.path.getsize(tsv_file_name) == 0:
        print(f"TSV file {tsv_file_name} is empty or missing.")
        return pd.DataFrame([])

    with open(tsv_file_name, "r") as file:
        lines = file.readlines()

    # Filter out comment lines and empty lines
    data_lines = [line for line in lines if not line.startswith("#") and line.strip()]

    # If no actual data after comments, return an empty array
    if not data_lines:
        print(f"TSV file {tsv_file_name} contains no actual data.")
        return pd.DataFrame([])

    # Try loading the file as a DataFrame
    try:
        data = pd.read_csv(tsv_file_name, sep="\t", comment="#", header=None)
        print("TSV file successfully parsed.")
        return data
    except pd.errors.EmptyDataError:
        print(f"TSV file {tsv_file_name} could not be read.")
        return pd.DataFrame([])


# Process Static Contacts Results
def process_vdw_rows(tsv_file_name: str, pdb_file: str) -> List[str]:
    
    vdw = load_tsv_data(tsv_file_name)
    #data = pd.read_csv(f"{tsv_file_name}", sep="\t", comment="#", header=None)
    ligand_resname = extract_ligand_name(pdb_file)
    
    if vdw.empty:  
        return []

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    ligand_first_atom_index = get_first_atom_index(structure, ligand_resname)

    results = []
    for _, row in vdw.iterrows():
        atom_1, atom_2, distance = row[2], row[3], float(row[4])
        chain_1, resname_1, resid_1, atom_name_1 = atom_1.split(":")
        chain_2, resname_2, resid_2, atom_name_2 = atom_2.split(":")

        if resname_1 == ligand_resname:
            ligand_atom_name = atom_name_1

            ligand_atom_index = get_atom_index(pdb_file, resname_1, ligand_atom_name) - ligand_first_atom_index
            protein_resid = int(resid_2) -1
            ligand_atom_type = atom_name_1[0]
        elif resname_2 == ligand_resname:
            ligand_atom_name = atom_name_2

            ligand_atom_index = get_atom_index(pdb_file, resname_2, ligand_atom_name) 
            ligand_first_atom_index = get_first_atom_index(structure, ligand_resname)
            print(f"🔍 Debugging in process_vdw_rows:")
            print(f"   - PDB File: {pdb_file}")
            print(f"   - Ligand Name: {ligand_resname}")
            print(f"   - Atom Name: {ligand_atom_name}")
            print(f"   - ligand_atom_index: {ligand_atom_index}")
            print(f"   - ligand_first_atom_index: {ligand_first_atom_index}")
            if ligand_atom_index is None:
                print(f"❌ Error: Could not find atom '{ligand_atom_name}' in ligand '{resname_2}' inside {pdb_file}")
                return []  # Skip processing this entry safely
            if ligand_first_atom_index is None:
                print(f"❌ Error: Could not determine first atom index for ligand '{resname_2}' in {pdb_file}")
                return []
            ligand_atom_index -= ligand_first_atom_index
            protein_resid = int(resid_1) -1
            ligand_atom_type = atom_name_2[0]
        else:
            continue

        results.append([0, ligand_atom_index, protein_resid, distance, ligand_atom_type, 0.000001])

    return results

def filter_vdw_overlaps(static_contacts, plip_results):
    """Filter out vdW interactions that overlap with hydrophobic interactions."""
    hydrophobic_pairs = set()
    filtered_results = []
    
    if not static_contacts:
        return []
    
    if not plip_results:
        return static_contacts

    # Extract hydrophobic interaction pairs from PLIP results
    for result in plip_results:
        if result[0] == 6:  # Hydrophobic interaction type is 6
            hydrophobic_pairs.add((result[1], result[2]))  # (ligand_atom_index, protein_resid)

    # Filter vdW interactions from static contacts results
    for result in static_contacts:
        if result[0] == 0:  # vdW interaction type is 0
            pair = (result[1], result[2])
            if pair not in hydrophobic_pairs:  # Keep if not overlapping
                filtered_results.append(result)

    return filtered_results

def cal_strength(results):
    if not results:
        return []
    for result in results:
        type_name = result[0]
        distance = float(result[3])  # Ensure distance is a float
        strength = 0.00001  # Default strength

        # vdW
        if type_name == 0:
            atom_type = result[4]
            if atom_type in ligand_vdw_params:
                req = ligand_vdw_params[atom_type]['Rmin/2'] * 2
                if req < distance:
                    strength = 1 - (distance - req) / (4.5 - req)
                elif req >= distance:
                    strength = 0.000001
        # H-bond
        elif type_name == 1:
            if 2.2 < distance < 4.1:
                strength = 1 - (distance - 2.2) / (4.1 - 2.2)
        # Salt bridge
        elif type_name == 2:
            if 2.8 < distance < 5.5:
                strength = 1 - (distance - 2.8) / (5.5 - 2.8)
        # Pi-stacking
        elif type_name == 3:
            if 3.4 < distance < 5.5:
                strength = 1 - (distance - 3.4) / (5.5 - 3.4)
        # Pi-cation
        elif type_name == 4:
            if 3.0 < distance < 6.0:
                strength = 1 - (distance - 3) / (6.0 -3)
        # Halogen bond
        elif type_name == 5:
            if 2.8 < distance < 4.0:
                strength = 1 - (distance - 2.8) / (4.0 - 2.8)
        # Hydrophobic
        elif type_name == 6:
            if 3 < distance < 5.0:
                strength = 1 - (distance - 3) / (5.0 - 3)

        result[5] = strength  # Clamp strength to [0, 1]
    return results


# File to persist custom IDs
CUSTOM_ID_FILE = "new_custom_ids.json"
CUSTOM_ID_PREFIX = "s"
custom_id_counter = 1  # Starting number for custom IDs

# Load existing custom IDs from the file if available
if os.path.exists(CUSTOM_ID_FILE):
    with open(CUSTOM_ID_FILE, "r") as f:
        custom_ids = json.load(f)
        custom_id_counter = max(
            int(id.replace(CUSTOM_ID_PREFIX, "")) for id in custom_ids.values()
        ) + 1
else:
    custom_ids = {}

def get_pubchem_id(smiles):
    global custom_ids, custom_id_counter

    # Check if this SMILES has already been assigned a custom ID
    if smiles in custom_ids:
        return custom_ids[smiles]

    # Step 1: Query PubChem for the SMILES
    compounds = get_compounds(smiles, namespace="smiles")
    if compounds and compounds[0].cid:
        return str(compounds[0].cid)  # Return PubChem ID if found

    # Step 2: Assign a new custom ID if not found in PubChem
    new_id = f"{CUSTOM_ID_PREFIX}{custom_id_counter}"
    custom_ids[smiles] = new_id
    custom_id_counter += 1

    # Persist the updated custom IDs to the file
    with open(CUSTOM_ID_FILE, "w") as f:
        json.dump(custom_ids, f, indent=4)

    return new_id

def get_uniprot_id(sequence):
    # File paths
    query_file = "query.fasta"
    db_path = "uniprot_sprot"  # Use the prefix of your BLAST database
    output_file = "results.xml"

    # Write the sequence to a FASTA file
    with open(query_file, "w") as f:
        f.write(f">query\n{sequence}\n")

    # Run BLAST
    blastp_command = [
        "blastp",
        "-query", query_file,
        "-db", db_path,
        "-out", output_file,
        "-outfmt", "5",  # XML format
        "-evalue", "0.001"
    ]
    result = subprocess.run(blastp_command, capture_output=True, text=True)

    # Check if BLAST ran successfully
    if result.returncode != 0:
        raise RuntimeError(f"Error running BLAST: {result.stderr}")

    # Parse results and extract the UniProt ID
    with open(output_file) as result_handle:
        blast_records = NCBIXML.read(result_handle)
        for alignment in blast_records.alignments:
            for hsp in alignment.hsps:
                description = alignment.title
                # Extract the real UniProt ID from the description
                if "sp|" in description:
                    uniprot_id = description.split("|")[3]  # Get the second field
                    return uniprot_id

    # If no matches are found, return None
    return None


#def assign_ids(results, ligand_pdb, protein_pdb):
#    pdb_to_selfies, selfies, smiles = pdb_to_selfies_mapping(ligand_pdb)
#    residue_to_saprot, saprot = map_residue_to_saprot(protein_pdb)
#    fasta = saprot_2_fasta(saprot)
#    for result in results:
#        result[6] = get_pubchem_id(smiles)
#        result[7] = get_uniprot_id(fasta)
#    return results


def pdb_to_selfies_mapping(pdb_file_path):
    mol = Chem.MolFromPDBFile(pdb_file_path, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError("Could not parse the PDB file.")

    # Generate SMILES and convert to SELFIES with atom mapping
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    original_smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
    selfies = sf.encoder(original_smiles)
    decoded_smiles, attr = sf.decoder(selfies, attribute=True)

    #original_tokens = re.findall(r"\[.*?\]", original_smiles)
    #decoded_tokens = re.findall(r"\[.*?\]", decoded_smiles)
    
    original_tokens = re.findall(r"\[.*?\]|\w|=", original_smiles)
    decoded_tokens = re.findall(r"\[.*?\]|\w|=", decoded_smiles)
    print(f"Original SMILES: {original_smiles}")  # Debug print
    print(f"Decoded SMILES: {decoded_smiles}")  # Debug print

    if len(original_tokens) != len(decoded_tokens):
        raise ValueError("Mismatch in token count between original and decoded SMILES.")
    original_to_decoded_mapping = {orig: dec for orig, dec in zip(original_tokens, decoded_tokens)}

    decoded_to_selfies_mapping = {}
    position_counter = {token: 0 for token in decoded_tokens}

    for entry in attr:
        smiles_token = entry.token
        for attrib in entry.attribution:
            if not attrib.token.startswith("[Branch"):
                selfies_index = attrib.index
                selfies_token = attrib.token
                token_position = position_counter[smiles_token]
                decoded_to_selfies_mapping[(smiles_token, token_position)] = (selfies_index, selfies_token)
                position_counter[smiles_token] += 1

    # Map PDB atoms to SELFIES tokens
    pdb_to_selfies_mappings = {}
    position_counter = {token: 0 for token in decoded_tokens}

    for token in original_tokens:
        #pdb_index = int(re.search(r":(\d+)", token).group(1))
        match = re.search(r":(\d+)", token)
        pdb_index = int(match.group(1)) if match else None  # Avoid AttributeError

        mapped_decoded_token = original_to_decoded_mapping[token]
        token_position = position_counter[mapped_decoded_token]
        selfies_data = decoded_to_selfies_mapping.get((mapped_decoded_token, token_position))
        if selfies_data:
            pdb_to_selfies_mappings[pdb_index] = {
                "SELFIES Token": selfies_data[1],
                "SELFIES Index": selfies_data[0],
            }
        else:
            print(f"Decoded token not found in SELFIES mapping: {mapped_decoded_token} at position {token_position}")
        position_counter[mapped_decoded_token] += 1

    return pdb_to_selfies_mappings, selfies, original_smiles

def map_residue_to_saprot(protein_path):

    # Parse the structure and extract sequences for all chains
    parsed_seqs = get_struc_seq(foldseek_path, protein_path, plddt_mask=False)

    # Initialize dictionaries to hold data for each chain
    residue_to_saprot_by_chain = {}
    combined_seq = ""

    for chain_id, (residues, sequence, saprot_seq) in parsed_seqs.items():
        # Split SaProt sequence into tokens (pairs of characters)
        saprot_tokens = [saprot_seq[i:i+2] for i in range(0, len(saprot_seq), 2)]

        # Map residues to SaProt tokens for this chain
        residue_to_saprot = {
            idx + 1: {"SaProt Token": token, "Residue Index": idx + 1}
            for idx, token in enumerate(saprot_tokens)
        }

        # Store data for this chain
        residue_to_saprot_by_chain[chain_id] = {
            "Residue to SaProt": residue_to_saprot,
            "Chain Sequence": saprot_seq
        }

        # Combine sequences for all chains
        combined_seq += saprot_seq

    return residue_to_saprot_by_chain, combined_seq


def saprot_selfies(ligand, protein):
    p_s_mapping, selfies_string = pdb_to_selfies_mapping(ligand)
    r_s_mapping, saprot_string = map_residue_to_saprot(protein)

    return selfies_string, saprot_string

def get_selfies_index(pdb_index):
    mapping = pdb_to_selfies.get(pdb_index)
    return mapping["SELFIES Index"]


#def convert_pdb_index_2_selfies_index(results):
#    for result in results:
#        ligand_pdb_index = result[1] + 1
#        selfiex_index = get_selfies_index(ligand_pdb_index)
#        result[1] = selfiex_index
#    return results

def convert_pdb_index_2_selfies_index(results, pdb_to_selfies):
    if not results:
        return []
    for result in results:
        ligand_pdb_index = result[1] + 1
        if ligand_pdb_index in pdb_to_selfies:
            selfies_index = pdb_to_selfies[ligand_pdb_index]["SELFIES Index"]
            result[1] = selfies_index
        else:
            print(f"Warning: Ligand PDB index {ligand_pdb_index} not found in pdb_to_selfies")
    return results


def saprot_2_fasta(saprot_sequence):
    return "".join([char for char in saprot_sequence if char.isupper()])

import numpy as np
import re

def get_attention_maps(results, ligand_pat, protein_pat):
    pdb_to_selfies, selfies, smiles = pdb_to_selfies_mapping(ligand_pat)
    residue_to_saprot, saprot = map_residue_to_saprot(protein_pat)
    fasta = saprot_2_fasta(saprot)
    res_number = len(fasta)
    token_number = len(re.findall(r'\[.*?\]', selfies))

    # Interaction types (7 specific + 1 total)
    interaction_types = [
        "vdw_interaction",
        "hydrogen_bond",
        "salt_bridge",
        "pi_stacking",
        "cation_pi_interaction",
        "halogen_bond",
        "hydrophobic_interactions",
        "total"
    ]

    # Initialize the attention map as a token_number x res_number x 8 matrix
    attention_map = np.full((token_number, res_number, 8), 1e-6)
    
    if not results:
        return attention_map  # Return empty initialized matrix if no results
    
    # Map interaction type indices
    interaction_map = {
        0: "vdw_interaction",
        1: "hydrogen_bond",
        2: "salt_bridge",
        3: "pi_stacking",
        4: "cation_pi_interaction",
        5: "halogen_bond",
        6: "hydrophobic_interactions"
    }

    # Populate attention maps
    for result in results:
        type_name = result[0]
        selfies_index = result[1]
        protein_index = result[2]
        strength = result[5]

        print(f"{selfies_index}, {protein_index}, {type_name}, {strength}")
        
        if type_name in interaction_map:
            idx = list(interaction_map.keys()).index(type_name)
            # Update only if the new strength is greater than the current value
            if strength > attention_map[selfies_index, protein_index, idx]:
                attention_map[selfies_index, protein_index, idx] = strength
    
    # Compute the total attention map (sum of first 7 layers, treating 1e-6 as 0)
    masked_attention = np.where(attention_map[:, :, :7] == 1e-6, 0, attention_map[:, :, :7])
    total_map = np.sum(masked_attention, axis=2)
    
    # Restore default value where total_map is 0
    attention_map[:, :, 7] = np.where(total_map == 0, 1e-6, total_map)
    
    return attention_map


import os
import json
import traceback
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import tempfile

def merge_protein_ligand(protein_file, ligand_file, output_file):
    cmd.set("retain_order", 1)

    try:
        cmd.load(protein_file, 'protein')
        cmd.load(ligand_file, 'ligand')
    except FileNotFoundError:
        print(f"Error: Could not find {protein_file} or {ligand_file}")
        return
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as protein_temp, \
         tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as ligand_temp:
        protein_temp_path = protein_temp.name
        ligand_temp_path = ligand_temp.name

        cmd.save(protein_temp_path, "protein")
        cmd.save(ligand_temp_path, "ligand")

    merged_content = []
    current_atom_index = 1
    current_residue_index = 1
    last_residue_number = None

    def process_pdb_file(input_file, current_atom_index, current_residue_index, last_residue_number):
        local_content = []
        with open(input_file, 'r') as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM", "TER")):
                    continue
                if line.startswith(("ATOM", "HETATM")):
                    residue_number = line[22:26].strip()
                    if residue_number != last_residue_number:
                        last_residue_number = residue_number
                        current_residue_index += 1
                    local_content.append(_renumber_line(line, current_atom_index, current_residue_index))
                    current_atom_index += 1
                else:
                    local_content.append(line)
        return local_content, current_atom_index, current_residue_index, last_residue_number

    merged_content, current_atom_index, current_residue_index, last_residue_number = \
        process_pdb_file(protein_temp_path, current_atom_index, current_residue_index, last_residue_number)

    ligand_content, current_atom_index, current_residue_index, last_residue_number = \
        process_pdb_file(ligand_temp_path, current_atom_index, current_residue_index, last_residue_number)

    merged_content.extend(ligand_content)

    with open(output_file, 'w') as f:
        f.writelines(merged_content)

    cmd.delete("protein")
    cmd.delete("ligand")
    cmd.delete("all")

def _renumber_line(line, new_atom_index, new_residue_index):
    atom_index = f"{new_atom_index:5d}"
    residue_index = f"{new_residue_index:4d}"
    return f"{line[:6]}{atom_index}{line[11:22]}{residue_index}{line[26:]}"

# --- Helper: Save a single complex as JSON ---
def save_complex_json(row, output_file):
    """
    Save a single protein–ligand complex result (a row) to a JSON file.

    Parameters:
        row (list): The k*12 matrix row for one complex.
        output_file (str): The path for the JSON file to save.
    """
    # Convert each attention map (or any numpy array) into a list so that it is JSON serializable.
    serializable_row = row[:4]  # First four columns (e.g., SELFIES, SAPROT, drug_id, protein_id)
    for attention_map in row[4:]:
        if isinstance(attention_map, np.ndarray):
            serializable_row.append(attention_map.tolist())
        else:
            serializable_row.append(attention_map)
    with open(output_file, "w") as f:
        json.dump(serializable_row, f)
    print(f"Saved complex JSON to {output_file}")


def extract_top_affinity(log_file):
    """
    Returns:
        float: Binding affinity (kcal/mol) or None if not found.
    """
    #log_file = f"{ligand}_{mode}_{trial}.log"  # Construct the log file name
    model = 1
    try:
        with open(log_file, 'r') as file:
            for line in file:
                if line.strip().startswith(f"{model}"):  # Match the model number
                    parts = line.split()
                    return float(parts[1])  # Extract the affinity value (2nd column)
    except FileNotFoundError:
        print(f"Warning: Log file {log_file} not found.")
    return None  # Return None if no affinity is found

# --- Worker Function: Process a single complex ---
def process_complex_task(task):
    """
    Worker function to process a single protein–ligand complex.

    Parameters:
        task (tuple): Contains (pdb_id, protein_index, protein_path, ligand_path, merged_path, output_dir).

    Returns:
        str or None: The output JSON file path if successful; otherwise, None.
    """
    pdb_id, protein_index, protein_path, ligand_path, merged_path, log_path, output_dir = task
    try:
        # Generate intermediate file names.
        output_tsv = os.path.join(output_dir, f"{pdb_id}_{protein_index}_interactions.tsv")
        output_xml = os.path.join(output_dir, f"{pdb_id}_{protein_index}_interactions.xml")
        output_file_name = f"{pdb_id}_{protein_index}_interactions"
        
        if not os.path.exists(merged_path):
            print(f"[{pdb_id} - {protein_index}] Merging files...")
            merge_protein_ligand(protein_path, ligand_path, merged_path)

        # Run intermediate steps if files do not exist.
        if not os.path.exists(output_tsv):
            print(f"[{pdb_id} - {protein_index}] Running get_static_contacts...")
            run_get_static_contacts(merged_path, output_tsv)
        else:
            print(f"[{pdb_id} - {protein_index}] TSV file already exists: {output_tsv}")

        if not os.path.exists(output_xml):
            print(f"[{pdb_id} - {protein_index}] Running PLIP...")
            run_plip(merged_path, output_file_name, output_dir)
        else:
            print(f"[{pdb_id} - {protein_index}] XML file already exists: {output_xml}")

        # Process the results.
        static_contacts_results = process_vdw_rows(output_tsv, merged_path)
        plip_results = process_file_pair(output_xml, merged_path)
        combined_results = cal_strength(static_contacts_results + plip_results)

        # Map and generate attention maps.
        pdb_to_selfies, selfies, smiles = pdb_to_selfies_mapping(ligand_path)
        residue_to_saprot, saprot = map_residue_to_saprot(protein_path)
        fasta = saprot_2_fasta(saprot)
        converted_results = convert_pdb_index_2_selfies_index(combined_results, pdb_to_selfies)
        maps = get_attention_maps(converted_results, ligand_path, protein_path)

        # Extract protein and drug IDs.
        drug_id = get_pubchem_id(smiles)
        protein_id = get_uniprot_id(fasta)
        energy = extract_top_affinity(log_path)   
        data_matrix = np.array([[selfies, saprot, drug_id, protein_id, energy, maps]], dtype=object)
        # Save this complex as its own npy file.
        complex_output_file = os.path.join(output_dir, f"{pdb_id}_{protein_index}_k12.npy")
        np.save(complex_output_file, data_matrix)
        print(f"[{pdb_id} - {protein_index}] Processed and saved complex.")
        return complex_output_file

    except Exception as e:
        print(f"Error processing {pdb_id} - Protein {protein_index}: {e}")
        traceback.print_exc()
        return None


# --- Task Preparation for a Single PDB ID ---
def prepare_tasks_for_pdb_id(pdb_id, processed_dir, merged_dir, output_dir):
    """
    Prepare a list of tasks (tuples) for a given PDB ID by matching corresponding protein, ligand, and merged files.

    Returns:
        list: A list of task tuples for this PDB ID.
    """
    tasks = []
    protein_files = [f for f in os.listdir(processed_dir)
                     if f.startswith(f"{pdb_id}_") and f.endswith("_protein_model1.pdb")]
    ligand_files = [f for f in os.listdir(processed_dir)
                    if f.startswith(f"{pdb_id}_") and f.endswith("_ligand_model1.pdb")]
    #merged_files = [f for f in os.listdir(merged_dir)
    #                if f.startswith(f"{pdb_id}_merged_") and f.endswith(".pdb")]
    log_files = [f for f in os.listdir(processed_dir)
                    if f.startswith(f"{pdb_id}_") and f.endswith(".log")]
    for protein_file in protein_files:
        protein_index = protein_file.split("_")[1].split(".")[0]
        matching_ligands = [f for f in ligand_files if f.split("_")[1].split(".")[0] == protein_index]
        #matching_merged = [f for f in merged_files if f.split("_")[-1].split(".")[0] == protein_index]
        matching_log = [f for f in log_files if f.split("_")[-1].split(".")[0] == protein_index]
        for ligand_file, log_file in zip(matching_ligands, matching_log):
            protein_path = os.path.join(processed_dir, protein_file)
            ligand_path = os.path.join(processed_dir, ligand_file)
            log_path = os.path.join(processed_dir, log_file)
            merged_path = os.path.join(merged_dir, f"{pdb_id}_merged_{protein_index}.pdb")
            if os.path.exists(protein_path) and os.path.exists(ligand_path) and os.path.exists(log_path):
                tasks.append((pdb_id, protein_index, protein_path, ligand_path, merged_path, log_path, output_dir))
            else:
                print(f"Skipping {pdb_id} - Protein {protein_index}: Missing required files.")
    return tasks


# --- Parallel Task Preparation ---
def prepare_all_tasks_parallel(txt_file, processed_dir, merged_dir, output_dir, num_cpu=None):
    """
    Prepare tasks for all PDB IDs listed in txt_file in parallel.

    Parameters:
        txt_file (str): Path to the text file containing PDB IDs.
        processed_dir (str): Directory with processed protein and ligand PDB files.
        merged_dir (str): Directory with merged PDB files.
        output_dir (str): Output directory (passed to tasks).
        num_cpu (int, optional): Number of CPU cores to use for task preparation.

    Returns:
        list: A flattened list of task tuples.
    """
    with open(txt_file, 'r') as file:
        pdb_ids = [line.strip() for line in file.readlines() if line.strip()]

    if num_cpu is None:
        num_cpu = mp.cpu_count()

    with mp.Pool(processes=num_cpu) as pool:
        tasks_per_pdb = pool.starmap(
            prepare_tasks_for_pdb_id,
            [(pdb_id, processed_dir, merged_dir, output_dir) for pdb_id in pdb_ids]
        )
    tasks = [task for sublist in tasks_per_pdb for task in sublist]
    return tasks


# --- Parallel Processing Function (with Task Preparation and Progress Display) ---
def process_all_pairs_to_k12_with_maps_parallel(txt_file, processed_dir, merged_dir, output_dir, num_cpu=None):
    """
    Process all protein–ligand pairs in parallel by first preparing tasks in parallel,
    then processing each complex in parallel, and generating individual k*12 JSON files.

    Parameters:
        txt_file (str): Path to the text file containing PDB IDs.
        processed_dir (str): Directory with processed protein and ligand PDB files.
        merged_dir (str): Directory with merged PDB files.
        output_dir (str): Directory to save intermediate outputs and individual JSON files.
        save_as (str): Format to save the complex result (currently only 'json' is implemented).
        num_cpu (int, optional): Number of CPU cores to use. Defaults to all available cores.

    Returns:
        list: A list of file paths for the saved JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare tasks in parallel.
    tasks = prepare_all_tasks_parallel(txt_file, processed_dir, merged_dir, output_dir, num_cpu)
    print(f"Prepared {len(tasks)} tasks from {len(open(txt_file).readlines())} PDB IDs.")

    if num_cpu is None:
        num_cpu = mp.cpu_count()
    print(f"Starting parallel processing with {num_cpu} CPUs on {len(tasks)} tasks...")

    # Process tasks in parallel with a progress bar.
    with mp.Pool(processes=num_cpu) as pool:
        results = list(tqdm(pool.imap(process_complex_task, tasks), total=len(tasks), desc="Processing complexes"))
    saved_npy_files = [r for r in results if r is not None]
    return saved_npy_files


# --- Main Execution ---
if __name__ == "__main__":
    # Define your parameters.
    #txt_file = "/work19/bai/baitokyotech/fusion_dock/docking/pdb_ids.txt"
    txt_file = "/work19/bai/baitokyotech/fusion_dock/docking/part_00.txt"
    processed_dir = "/work19/bai/baitokyotech/fusion_dock/docking/output_dataset"
    merged_dir = "/work19/bai/baitokyotech/fusion_dock/docking/complete_merged_pdbs"
    output_dir = "/work19/bai/baitokyotech/fusion_dock/docking/complete_output_npy/"

    # Set the number of CPUs you want to use (for example, 2).
    num_cpu = 20

    # Call the parallel processing function.
    saved_files = process_all_pairs_to_k12_with_maps_parallel(
        txt_file=txt_file,
        processed_dir=processed_dir,
        merged_dir=merged_dir,
        output_dir=output_dir,
        num_cpu=num_cpu
    )

    print("Parallel processing completed. The following JSON files were saved:")
    for f in saved_files:
        print(f)

