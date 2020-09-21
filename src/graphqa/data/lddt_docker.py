#! /usr/bin/python3
"""
Run LDDT from within script.

This script is meant to run inside a docker container with OpenStructure installed,
e.g. registry.scicore.unibas.ch/schwede/openstructure:2.1.0.

The paths on the host system pointing to the native structure, the decoys dir, and the output file
should be mounted under /mnt inside the container.
"""

import argparse
from pathlib import Path

import numpy as np
from ost.io import LoadPDB
from ost.mol.alg import (CleanlDDTReferences,
                         PreparelDDTGlobalRDMap,
                         lDDTSettings,
                         CheckStructure,
                         LocalDistDiffTest,
                         GetlDDTPerResidueStats,
                         PrintlDDTPerResidueStats,
                         ResidueNamesMatch)
from ost.io import ReadStereoChemicalPropsFile

structural_checks = False
bond_tolerance = 12
angle_tolerance = 12
cutoffs = [0.5, 1.0, 2.0, 4.0]

# Initialize settings with default parameters and print them
settings = lDDTSettings()
# settings.PrintParameters()

parser = argparse.ArgumentParser(description='LDDT')
parser.add_argument('seq_len', type=int)
parser.add_argument('native_pdb')
parser.add_argument('decoy_dir')
parser.add_argument('output_file')
args = parser.parse_args()

seq_len = args.seq_len
native_path = Path('/native') / args.native_pdb
decoy_dir = Path('/decoy') / args.decoy_dir
output_file = Path('/output') / args.output_file

# Prepare references - it should be a list of EntityView(s)
references = [LoadPDB(native_path.resolve().as_posix()).CreateFullView()]

# Clean up references
CleanlDDTReferences(references)

# Prepare residue map from references
rdmap = PreparelDDTGlobalRDMap(references,
                               cutoffs=cutoffs,
                               sequence_separation=settings.sequence_separation,
                               radius=settings.radius)

# Load model and prepare its view
decoy_paths = sorted(decoy_dir.glob('*.pdb'))

lddt_global_scores = np.full(len(decoy_paths), fill_value=np.nan)
lddt_local_scores = np.full((len(decoy_paths), seq_len), fill_value=np.nan)

for model_idx, model_path in enumerate(decoy_paths):
    model = LoadPDB(model_path.resolve().as_posix())
    model_view = model.GetChainList()[0].Select("peptide=true")
    
    # This part is optional and it depends on our settings parameter
    if structural_checks:
        stereochemical_parameters = ReadStereoChemicalPropsFile()
        CheckStructure(ent=model_view,
                    bond_table=stereochemical_parameters.bond_table,
                    angle_table=stereochemical_parameters.angle_table,
                    nonbonded_table=stereochemical_parameters.nonbonded_table,
                    bond_tolerance=bond_tolerance,
                    angle_tolerance=angle_tolerance)

    # Check consistency
    is_cons = ResidueNamesMatch(model_view, references[0], True)
    if not is_cons:
        print("Consistency check failed!")

    # Calculate lDDT
    LocalDistDiffTest(model_view,
                    references,
                    rdmap,
                    settings)

    # Get the local scores
    local_scores = GetlDDTPerResidueStats(model_view,
                                        rdmap,
                                        structural_checks,
                                        settings.label)

    # Print local scores
    # PrintlDDTPerResidueStats(local_scores, structural_checks, len(cutoffs))

    # Compute preserved contact
    conserved = np.full(seq_len, fill_value=np.nan)
    total = np.full(seq_len, fill_value=np.nan)
    for residue in local_scores:
        if residue.is_assessed == 'Yes':
            conserved[residue.rnum - 1] = residue.conserved_dist
            total[residue.rnum - 1] = residue.total_dist
    local_lddt = conserved / total
    global_lddt = np.nansum(conserved) / np.nansum(total)
    
    # Store
    lddt_global_scores[model_idx] = global_lddt
    lddt_local_scores[model_idx, :] = local_lddt

np.savez(output_file.resolve(), **{
    'decoys': np.array([p.with_suffix('').name for p in decoy_paths]),
    'global_lddt': lddt_global_scores,
    'local_lddt': lddt_local_scores,
})
