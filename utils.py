import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
import pandas as pd

def get_molecule_data(cid):
    """
    Получение свойств молекулы по ее CID из PubChem.
    
    Parameters:
    cid (int): CID молекулы
    
    Returns:
    dict: Словарь с рассчитанными свойствами
    """
    compound = pcp.Compound.from_cid(cid)
    return {
        'cid': cid,
        'Molecule (RDKit Mol)': compound.isomeric_smiles,
        'SlogP': compound.xlogp if compound else None,
        'SMR': 'unknown',
        'TPSA': compound.tpsa if compound else None,
        'AMW': compound.molecular_weight if compound else None,
        'ExactMW': compound.exact_mass if compound else None,
        'NumLipinskiHBA': compound.h_bond_acceptor_count if compound else None,
        'NumLipinskiHBD': compound.h_bond_donor_count if compound else None,
        'NumRotatableBonds': compound.rotatable_bond_count if compound else None,
        'NumHeavyAtoms': compound.heavy_atom_count if compound else None,
    }

def calculate_mqn(smiles):
    """
    Рассчитать Molecular Quantum Numbers (MQN) для заданной молекулы.
    
    Parameters:
    smiles (str): SMILES-строка молекулы
    
    Returns:
    list: Список из 42 MQN
    """
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.MQNs_(mol) if mol else [None] * 42

def calculate_properties(smiles):
    """
    Рассчитать молекулярные свойства для заданной молекулы через RDKit.
    
    Parameters:
    smiles (str): SMILES-строка молекулы
    
    Returns:
    dict: Словарь с рассчитанными свойствами
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            properties = {
                'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(mol),
                'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(mol),
                'NumAmideBonds': int(rdMolDescriptors.CalcNumAmideBonds(mol)),
                'NumHeteroAtoms': int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
                'NumAtoms': int(mol.GetNumAtoms()),
                'NumRings': int(rdMolDescriptors.CalcNumRings(mol)),
                'NumAromaticRings': int(rdMolDescriptors.CalcNumAromaticRings(mol)),
                'NumSaturatedRings': int(rdMolDescriptors.CalcNumSaturatedRings(mol)),
                'NumAliphaticRings': int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
                'NumAromaticHeterocycles': int(rdMolDescriptors.CalcNumAromaticHeterocycles(mol)),
                'NumSaturatedHeterocycles': int(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)),
                'NumAliphaticHeterocycles': int(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)),
                'NumAromaticCarbocycles': int(rdMolDescriptors.CalcNumAromaticCarbocycles(mol)),
                'NumSaturatedCarbocycles': int(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)),
                'NumAliphaticCarbocycles': int(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)),
                'HallKierAlpha': rdMolDescriptors.CalcHallKierAlpha(mol),
                'kappa1': rdMolDescriptors.CalcKappa1(mol),
                'kappa2': rdMolDescriptors.CalcKappa2(mol),
                'kappa3': rdMolDescriptors.CalcKappa3(mol),
                'Chi0v': rdMolDescriptors.CalcChi0v(mol),
                'Chi1v': rdMolDescriptors.CalcChi1v(mol),
                'Chi2v': rdMolDescriptors.CalcChi2v(mol),
                'Chi3v': rdMolDescriptors.CalcChi3v(mol),
                'Chi4v': rdMolDescriptors.CalcChi4v(mol),
                'Chi1n': rdMolDescriptors.CalcChi1n(mol),
                'Chi2n': rdMolDescriptors.CalcChi2n(mol),
                'Chi3n': rdMolDescriptors.CalcChi3n(mol),
                'Chi4n': rdMolDescriptors.CalcChi4n(mol),
                'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
                'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol) 
            }
            # Автоматически собираем названия свойств
            properties_names = list(properties.keys())
            return properties
        else:
            return {prop: None for prop in properties_names}
    except:
        return {prop: None for prop in properties_names}

def calculate_vsa_descriptors(smiles):
    """
    Рассчитать дескрипторы VSA (VolSurf-like surface area) для заданной молекулы через RDKit.

    Дескрипторы включают:
    - SlogP_VSA (VolSurf area descriptors based on SlogP)
    - SMR_VSA (VolSurf area descriptors based on SMR)
    - PEOE_VSA (VolSurf area descriptors based on partial equalization of orbital electronegativity)

    Parameters:
    smiles (str): SMILES-строка молекулы
    
    Returns:
    dict: Словарь с рассчитанными VSA-дескрипторами (SlogP_VSA, SMR_VSA, PEOE_VSA)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Формируем словарь VSA дескрипторов
            vsa_properties = {
                'slogp_VSA1': rdMolDescriptors.SlogP_VSA_(mol)[0],
                'slogp_VSA2': rdMolDescriptors.SlogP_VSA_(mol)[1],
                'slogp_VSA3': rdMolDescriptors.SlogP_VSA_(mol)[2],
                'slogp_VSA4': rdMolDescriptors.SlogP_VSA_(mol)[3],
                'slogp_VSA5': rdMolDescriptors.SlogP_VSA_(mol)[4],
                'slogp_VSA6': rdMolDescriptors.SlogP_VSA_(mol)[5],
                'slogp_VSA7': rdMolDescriptors.SlogP_VSA_(mol)[6],
                'slogp_VSA8': rdMolDescriptors.SlogP_VSA_(mol)[7],
                'slogp_VSA9': rdMolDescriptors.SlogP_VSA_(mol)[8],
                'slogp_VSA10': rdMolDescriptors.SlogP_VSA_(mol)[9],
                'slogp_VSA11': rdMolDescriptors.SlogP_VSA_(mol)[10],
                'slogp_VSA12': rdMolDescriptors.SlogP_VSA_(mol)[11],
                'smr_VSA1': rdMolDescriptors.SMR_VSA_(mol)[0],
                'smr_VSA2': rdMolDescriptors.SMR_VSA_(mol)[1],
                'smr_VSA3': rdMolDescriptors.SMR_VSA_(mol)[2],
                'smr_VSA4': rdMolDescriptors.SMR_VSA_(mol)[3],
                'smr_VSA5': rdMolDescriptors.SMR_VSA_(mol)[4],
                'smr_VSA6': rdMolDescriptors.SMR_VSA_(mol)[5],
                'smr_VSA7': rdMolDescriptors.SMR_VSA_(mol)[6],
                'smr_VSA8': rdMolDescriptors.SMR_VSA_(mol)[7],
                'smr_VSA9': rdMolDescriptors.SMR_VSA_(mol)[8],
                'smr_VSA10': rdMolDescriptors.SMR_VSA_(mol)[9],
                'peoe_VSA1': rdMolDescriptors.PEOE_VSA_(mol)[0],
                'peoe_VSA2': rdMolDescriptors.PEOE_VSA_(mol)[1],
                'peoe_VSA3': rdMolDescriptors.PEOE_VSA_(mol)[2],
                'peoe_VSA4': rdMolDescriptors.PEOE_VSA_(mol)[3],
                'peoe_VSA5': rdMolDescriptors.PEOE_VSA_(mol)[4],
                'peoe_VSA6': rdMolDescriptors.PEOE_VSA_(mol)[5],
                'peoe_VSA7': rdMolDescriptors.PEOE_VSA_(mol)[6],
                'peoe_VSA8': rdMolDescriptors.PEOE_VSA_(mol)[7],
                'peoe_VSA9': rdMolDescriptors.PEOE_VSA_(mol)[8],
                'peoe_VSA10': rdMolDescriptors.PEOE_VSA_(mol)[9],
                'peoe_VSA11': rdMolDescriptors.PEOE_VSA_(mol)[10],
                'peoe_VSA12': rdMolDescriptors.PEOE_VSA_(mol)[11],
                'peoe_VSA13': rdMolDescriptors.PEOE_VSA_(mol)[12],
                'peoe_VSA14': rdMolDescriptors.PEOE_VSA_(mol)[13]
            }
            
            # Автоматически собираем названия дескрипторов
            vsa_property_names = list(vsa_properties.keys())
            return vsa_properties
        else:
            return {prop: None for prop in vsa_property_names}
    except:
        return {prop: None for prop in vsa_property_names}


def recalculate_slogp_for_missing(smiles, slogp):
    """
    Пересчитать SlogP для молекул, у которых отсутствует это значение.
    
    Parameters:
    smiles (str): SMILES-строка молекулы
    slogp (float or None): Исходное значение SlogP
    
    Returns:
    float or None: Рассчитанное значение SlogP, если возможно
    """
    if pd.notnull(slogp):
        return slogp
    mol = Chem.MolFromSmiles(smiles)
    return Crippen.MolLogP(mol) if mol else None