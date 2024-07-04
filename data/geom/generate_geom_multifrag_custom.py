import argparse
import json
import numpy as np
import pandas as pd
import re
import random

from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.BRICS import FindBRICSBonds
from tqdm import tqdm
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem

# #################################################################################### #
# ####################################### BRICS ###################################### #
# #################################################################################### #
REGEX = re.compile('\[\d*\*\]')


def split_into_n_fragments(mol, bonds, num_frags):
    num_bonds = num_frags - 1
    bondidx2minfrag = {}
    bondidx2atoms = {}
    for bond in bonds:
        bond_idx = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx()
        frags = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
        for atom in frags.GetAtoms():
            atom.SetIsAromatic(False)
        frags = Chem.GetMolFrags(frags, asMols=True)
        minfragsize = min([f.GetNumAtoms() for f in frags])
        bondidx2minfrag[bond_idx] = minfragsize
        bondidx2atoms[bond_idx] = bond

    # Selecting only top-N bonds connecting the biggest 6 fragments
    sorted_bonds = sorted(bondidx2minfrag.keys(), key=lambda bidx: -bondidx2minfrag[bidx])
    bonds_to_split = sorted_bonds[:num_bonds]

    # Selecting atoms connected by top-N bonds
    # Note that we add 1 to start numeration from 1 to correctly assign labels (RDKit issue)
    bond_atoms = [
        (bondidx2atoms[bidx][0] + 1, bondidx2atoms[bidx][1] + 1)
        for bidx in bonds_to_split
    ]

    frags = Chem.FragmentOnBonds(mol, bonds_to_split, addDummies=True, dummyLabels=bond_atoms)
    for atom in frags.GetAtoms():
        atom.SetIsAromatic(False)
    frags = Chem.GetMolFrags(frags, asMols=True)
    return frags, bond_atoms


def check_fragments_brics(frags, min_frag_size):
    for frag in frags:
        num_dummy_atoms = len(re.findall(REGEX, Chem.MolToSmiles(frag)))
        if (frag.GetNumAtoms() - num_dummy_atoms) < min_frag_size:
            return False

    return True


def generate_possible_connected_linkers(neighbors):
    candidates = np.where(neighbors.sum(0) > 2)[0]
    possible_linkers = set([
        (candidate,)
        for candidate in candidates
    ])
    return possible_linkers


def generate_possible_2nd_order_linkers(neighbors):
    # Removing edge fragments
    initial_candidates = np.where(neighbors.sum(0) > 1)[0]
    neighbors = neighbors[initial_candidates][:, initial_candidates]

    # Computing 2nd order neighbors and finding all loops
    second_order_neigh = ((neighbors @ neighbors) > 0).astype(int) * (1 - neighbors) - np.eye(neighbors.shape[0])
    candidates = set(np.where(np.diag(second_order_neigh @ second_order_neigh))[0])

    possible_linkers_pairs = set()
    for first_candidate in candidates:
        for second_candidate in set(np.where(second_order_neigh[first_candidate])[0]) & candidates:
            linker_1 = initial_candidates[first_candidate]
            linker_2 = initial_candidates[second_candidate]
            if linker_1 != linker_2:
                possible_linkers_pairs.add(tuple(sorted([linker_1, linker_2])))

    return possible_linkers_pairs


def generate_possible_3nd_order_linkers(neighbors):
    # Removing edge fragments
    initial_candidates = np.where(neighbors.sum(0) > 1)[0]
    neighbors = neighbors[initial_candidates][:, initial_candidates]

    # Computing 3rd order neighbors and finding all loops
    third_order_neigh = ((neighbors @ neighbors @ neighbors) > 0).astype(int)
    third_order_neigh = third_order_neigh * (1 - neighbors) - np.eye(neighbors.shape[0])
    candidates = set(np.where(np.diag(third_order_neigh @ third_order_neigh @ third_order_neigh))[0])

    possible_linkers_triples = set()
    for first_candidate in candidates:
        rest_candidates = candidates.difference({first_candidate})
        rest_candidates = set(np.where(third_order_neigh[first_candidate])[0]) & rest_candidates
        for second_candidate in rest_candidates:
            remainders = rest_candidates.difference({second_candidate})
            for third_candidate in remainders:
                linker_1 = initial_candidates[first_candidate]
                linker_2 = initial_candidates[second_candidate]
                linker_3 = initial_candidates[third_candidate]
                if linker_1 != linker_2 and linker_1 != linker_3 and linker_2 != linker_3:
                    possible_linkers_triples.add(tuple(sorted([linker_1, linker_2, linker_3])))

    return possible_linkers_triples


def fragment_by_brics(smiles, min_frag_size, max_frag_size, num_frags, linker_type=None):
    mol = Chem.MolFromSmiles(smiles)

    # Finding all BRICS bonds
    bonds = [bond[0] for bond in FindBRICSBonds(mol)]
    if len(bonds) == 0:
        return []

    # Splitting molecule into fragments
    frags, bond_atoms = split_into_n_fragments(mol, bonds, num_frags)
    if not check_fragments_brics(frags, min_frag_size):
        return []

    # Building mapping between fragments and connecting atoms
    atom2frag = {}
    for i, frag in enumerate(frags):
        matches = re.findall(REGEX, Chem.MolToSmiles(frag))
        for match in matches:
            atom = int(match[1:-2])
            atom2frag[atom] = i

    # Creating adjacency matrix
    neighbors = np.zeros((len(frags), len(frags)))
    for atom1, atom2 in bond_atoms:
        neighbors[atom2frag[atom1], atom2frag[atom2]] = 1
        neighbors[atom2frag[atom2], atom2frag[atom1]] = 1

    # Generating possible linkers
    if linker_type is None:
        possible_linkers = []
        possible_linkers += generate_possible_connected_linkers(neighbors)
        possible_linkers += generate_possible_2nd_order_linkers(neighbors)
        possible_linkers += generate_possible_3nd_order_linkers(neighbors)
    elif linker_type == 1:
        possible_linkers = generate_possible_connected_linkers(neighbors)
    elif linker_type == 2:
        possible_linkers = generate_possible_2nd_order_linkers(neighbors)
    elif linker_type == 3:
        possible_linkers = generate_possible_3nd_order_linkers(neighbors)
    else:
        raise NotImplementedError

    # Formatting the results
    results = []
    for linkers in possible_linkers:
        linkers_smi = ''
        fragments_smi = ''
        for i in range(len(frags)):
            if i in linkers:
                linkers_smi += Chem.MolToSmiles(frags[i]) + '.'
            else:
                fragments_smi += Chem.MolToSmiles(frags[i]) + '.'

        linkers_smi = linkers_smi[:-1]
        fragments_smi = fragments_smi[:-1]
        results.append([smiles, linkers_smi, fragments_smi, 'brics'])

    return results


# #################################################################################### #
# ####################################### BIND SITE ################################## #
# #################################################################################### #
# 函数：识别氢键给体和受体
def find_h_bond_donors_and_acceptors(mol):
    h_bond_donors = []
    h_bond_acceptors = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # H atom
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() in [7, 8]:  # N, O
                    h_bond_donors.append(atom.GetIdx())
        elif atom.GetAtomicNum() in [7, 8]:  # N, O
            if Lipinski.NumHAcceptors(mol) > 0:
                h_bond_acceptors.append(atom.GetIdx())

    return h_bond_donors, h_bond_acceptors


# 函数：识别疏水基团
def find_hydrophobic_groups(mol):
    hydrophobic_groups = []
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic() and atom.GetAtomicNum() == 6:  # Non-aromatic carbon atoms
            hydrophobic_groups.append(atom.GetIdx())
    return hydrophobic_groups


# 函数：识别芳香基团
def find_aromatic_groups(mol):
    aromatic_groups = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_groups.append(atom.GetIdx())
    return aromatic_groups


# 函数：识别静电相互作用
def find_electrostatic_interactions(mol):
    positive_sites = []
    negative_sites = []

    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            positive_sites.append(atom.GetIdx())
        elif atom.GetFormalCharge() < 0:
            negative_sites.append(atom.GetIdx())

    return positive_sites, negative_sites


# 函数：识别潜在配位键部位
def find_coordination_sites(mol):
    coordination_sites = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8, 16]:  # N, O, S
            coordination_sites.append(atom.GetIdx())
    return coordination_sites


def get_break_bonds(molecule, atom_indices):
    """
    打断选定原子和其它原子的链接，形成片段
    :param molecule: RDKit 分子对象
    :param atom_indices: 要截断的原子的索引列表
    :return: bonds: 要打断的键列表
    """
    # 记录需要打断的键 (由所选原子涉及的所有键)
    bonds_to_break = []
    for idx in atom_indices:
        atom = molecule.GetAtomWithIdx(idx)
        bonds = atom.GetBonds()
        for bond in bonds:
            bonds_to_break.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return bonds_to_break


# sort bind site atom
def sort_bind_atom(mol, h_bond_donors, h_bond_acceptors, hydrophobic_groups, positive_sites, negative_sites, coordination_sites):
    all_idx = list(set(h_bond_donors + h_bond_acceptors + hydrophobic_groups + positive_sites +
                       negative_sites + coordination_sites))
    score = {i: 0 for i in all_idx}
    # 不同作用的得分不同
    for idx in all_idx:
        if idx in h_bond_donors or idx in h_bond_acceptors:
            score[idx] += 4
        if idx in hydrophobic_groups:
            score[idx] += 3
        if idx in positive_sites or idx in negative_sites:
            score[idx] += 2
        if idx in coordination_sites:
            score[idx] += 1
    # 根据原子的连边数量，赋予得分。一般在外围的原子连边少，得分高
    # 获取该原子连接的原子数目
    for idx in all_idx:
        atom = mol.GetAtomWithIdx(idx)
        degree = atom.GetDegree()
        if degree == 1:
            score[idx] += 5
        elif degree == 2:
            score[idx] += 3
        elif degree == 3:
            score[idx] += 1
    if len(score) <= 10:
        return list(score.keys())
    else:
        score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
        return list(score.keys())[:10]


# #################################################################################### #
# ####################################### Main ####################################### #
# #################################################################################### #

def run(geom_json_path, output_path):
    with open(geom_json_path) as f:
        geom_json = json.load(f)

    all_smiles = list(geom_json.keys())
    mol_results = []
    sample_numbers = {3: 5, 4: 5, 5: 3, 6: 3, 7: 1, 8: 1, 9: 1, 10: 1}

    for i, smiles in tqdm(enumerate(all_smiles), total=len(all_smiles)):
        if i == 1000:
            break
        mol = Chem.MolFromSmiles(smiles)

        # 进行完整的规范化和检查
        AllChem.Kekulize(mol, clearAromaticFlags=True)

        # 识别不同类型的相互作用位点
        h_bond_donors, h_bond_acceptors = find_h_bond_donors_and_acceptors(mol)
        hydrophobic_groups = find_hydrophobic_groups(mol)
        # aromatic_groups = find_aromatic_groups(mol)
        positive_sites, negative_sites = find_electrostatic_interactions(mol)
        coordination_sites = find_coordination_sites(mol)

        # 候选结合原子
        candidate_atoms = sort_bind_atom(mol, h_bond_donors, h_bond_acceptors, hydrophobic_groups, positive_sites,
                                         negative_sites, coordination_sites)

        # 采样需要分离的原子索引
        atom_indexes = []
        for k in range(3, min(11, len(candidate_atoms) + 1)):
            for t in range(sample_numbers[k]):
                atom_index = list(set(random.sample(candidate_atoms, k)))
                # atom_index = [0, 4, 6]
                if atom_index not in atom_indexes:
                    atom_indexes.append(atom_index)

        # 根据需要打断的键，生成片段smile
        for atom_index in atom_indexes:
            bonds_to_break = get_break_bonds(mol, atom_index)

            for a in mol.GetAtoms():
                a.SetIntProp("__origIdx", a.GetIdx())

            frags, bond_atoms = split_into_n_fragments(mol, bonds_to_break, 100)

            linkers = []
            for i, frag in enumerate(frags):
                num_dummy_atoms = len(re.findall(REGEX, Chem.MolToSmiles(frag)))
                if (frag.GetNumAtoms() - num_dummy_atoms) == 1 and int(
                        frag.GetAtoms()[0].GetProp('__origIdx')) in atom_index:
                    pass
                else:
                    linkers.append(i)
            # Formatting the results
            linkers_smi = ''
            fragments_smi = ''
            for i in range(len(frags)):
                if i in linkers:
                    linkers_smi += Chem.MolToSmiles(frags[i]) + '.'
                else:
                    fragments_smi += Chem.MolToSmiles(frags[i]) + '.'

            linkers_smi = linkers_smi[:-1]
            fragments_smi = fragments_smi[:-1]
            mol_results.append([smiles, linkers_smi, fragments_smi, 'brics'])

        # if (i + 1) % 5000 == 0:
        #     table = pd.DataFrame(mol_results, columns=['molecule', 'linker', 'fragments', 'method'])
        #     table = table.drop_duplicates(['molecule', 'linker'])
        #     table.to_csv(output_path, index=False)

    table = pd.DataFrame(mol_results, columns=['molecule', 'linker', 'fragments', 'method'])
    table = table.drop_duplicates(['molecule', 'linker'])
    table.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geom-json', action='store', type=str, required=True)
    parser.add_argument('--output', action='store', type=str, required=True)
    args = parser.parse_args()

    run(
        geom_json_path=args.geom_json,
        output_path=args.output,
    )
