# Hacky?


class Input(object):
    class Node(object):
        RESIDUES = slice(0, 22)
        PARTIAL_ENTROPY = slice(RESIDUES.stop, RESIDUES.stop + 23)
        SELF_INFO = slice(PARTIAL_ENTROPY.stop, PARTIAL_ENTROPY.stop + 23)
        DSSP_FEATURES = slice(SELF_INFO.stop, SELF_INFO.stop + 15)

        DSSP_3_STATE_OH = slice(DSSP_FEATURES.start, DSSP_FEATURES.start + 3)
        DSSP_6_STATE_OH = slice(DSSP_3_STATE_OH.stop, DSSP_3_STATE_OH.stop + 6)
        DSSP_SURFACE_ACCESSIBILITY = DSSP_FEATURES.start + 9
        DSSP_PHI_SIN = DSSP_FEATURES.start + 10
        DSSP_PHI_COS = DSSP_FEATURES.start + 11
        DSSP_PSI_SIN = DSSP_FEATURES.start + 12
        DSSP_PSI_COS = DSSP_FEATURES.start + 13
        DSSP_VALID = DSSP_FEATURES.start + 14

        # Missing residues in multi-sequence alignments are represented as '_'
        AMINOACID_NAMES = tuple('ACDEFGHIKLMNOPQRSTUVWY_')
        DSSP_3_STATE_NAMES = tuple('HEC')
        DSSP_3_STATE_NAMES = tuple('HEC')
        DSSP_6_STATE_NAMES = tuple('GI H E B TS C'.split())

        LENGTH = 83

    class Edge(object):
        SPATIAL_DISTANCE = 0
        IS_PEPTIDE_BOND = 1
        SEPARATION_OH = slice(1, 1 + 7)

        LENGTH = 8


class Output(object):
    class Node(object):
        LOCAL_LDDT = 0
        LOCAL_LDDT_WEIGHT = 1

        LENGTH = 1

    class Global(object):
        GLOBAL_LDDT = 0
        GLOBAL_GDTTS = 1
        GLOBAL_LDDT_WEIGHT = 2
        GLOBAL_GDTTS_WEIGHT = 3

        LENGTH = 2
