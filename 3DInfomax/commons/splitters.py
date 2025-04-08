
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger  


def generate_scaffold(smiles, include_chirality=False):
    """ Obtain Bemis-Murcko scaffold from smiles
    :return: smiles of scaffold """

    RDLogger.DisableLog('rdApp.*') #disables warnings

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


