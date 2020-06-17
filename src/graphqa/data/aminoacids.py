aa_1_mapping = {
    # 20 common amino acids
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
    # Rare ones (replace U->S, O->K)
    'U': 15,
    'O': 11,
    # Ambiguous cases
    'B': 3, # B -> D, N
    'Z': 6, # Z -> E, Q
    'J': 9, # J -> I, L
}

aa_1_mapping_inv = [
    'A',
    'R',
    'N',
    'D',
    'C',
    'Q',
    'E',
    'G',
    'H',
    'I',
    'L',
    'K',
    'M',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'V',
]

aa_3_mapping = {
    # 20 common amino acids
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLN': 5,
    'GLU': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19,
    # Rare ones (replace SEC->SER, PYR->LYS, MSE->MET)
    'SEC': 15,
    'PYR': 11,
    'MSE': 12
}

aa_3_mapping_inv = [
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL',
]