# Hacky?


class Input(object):
    class Node(object):
        RESIDUES = slice(0, 22)
        PARTIAL_ENTROPY = slice(RESIDUES.stop, RESIDUES.stop + 23)
        SELF_INFO = slice(PARTIAL_ENTROPY.stop, PARTIAL_ENTROPY.stop + 23)
        DSSP_FEATURES = slice(SELF_INFO.stop, SELF_INFO.stop + 15)

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
        NATIVE_DETERMINED = 2

        LENGTH = 1

    class Global(object):
        GLOBAL_LDDT = 0
        GLOBAL_GDTTS = 1
        GLOBAL_LDDT_WEIGHT = 2
        GLOBAL_GDTTS_WEIGHT = 3

        LENGTH = 2
