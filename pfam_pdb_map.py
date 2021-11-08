# Uses MIToS (https://docs.juliahub.com/MIToS/PuGh5/2.6.1/Pfam/#Module-Pfam) to map from PFAM MSA sequences to PDB residue positions

from Bio.PDB import *
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBList import PDBList
from Bio import pairwise2
from Bio import SeqIO




# sequence-alignment from https://www.biostars.org/p/10094/
def resmap(chain, uniprot_sequence):
# Returns a PDB to UniProt residue number dictionary. 

    ppb=PPBuilder()
    polypeptides = ppb.build_peptides(chain)
    pdb_sequence = ""
    for polypeptide in polypeptides:
        pdb_sequence = pdb_sequence + polypeptide.get_sequence()
    pdb_res_nums = sortedres.id[1] for res in chain if res.id[0] == " ")

    residue_list = Selection.unfold_entities(chain, 'R')
    alignments = pairwise2.align.globalms(uniprot_sequence, pdb_sequence, 2, -1, -.5, -.1)    
    uniprot_align = str(alignments[0][0])
    pdb_align     = str(alignments[0][1])

    uniprot_map = []
    count = 0
    for residue in uniprot_align:
        if residue != "-":
            count += 1
            uniprot_map.append(count)
        else:
            uniprot_map.append(-1)

    pdb_map = []
    count = -1
    for residue in pdb_align:
        if residue != "-":
            count += 1
            pdb_map.append(pdb_res_nums[count])
        else:
            pdb_map.append(-1)

    matches = []
    for index, residue in enumerate(uniprot_map):
        if uniprot_align[index] == pdb_align[index] and uniprot_align[index] != "-" and pdb_align[index] != "-":
            matches.append(True)
        else:
            matches.append(False)

    mapping = {}
    for index, match in enumerate(matches):
        if match:
            mapping[pdb_map[index]] = uniprot_map[index]

    return mapping
