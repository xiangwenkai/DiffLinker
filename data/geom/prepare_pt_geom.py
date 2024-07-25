import argparse
import json
import time

import numpy as np
import pandas as pd
import re
import os
import pickle
import random
import torch
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
    if mol is None:
        return "Invalid SMILES"

    num_valence_electrons = {
        'H': 1, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 7, 'I': 7
    }

    donors = []
    acceptors = []
    donor_acceptors = []

    # 遍历分子中的每个原子
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_num = atom.GetAtomicNum()
        num_hydrogens = atom.GetTotalNumHs()

        # 计算孤对电子数量
        # 实际参与的共价键数（键序和）
        total_bonds = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])
        # 原子的形式电荷
        formal_charge = atom.GetFormalCharge()
        # 计算孤对电子数
        lone_pairs = (num_valence_electrons[atom.GetSymbol()] - total_bonds - formal_charge) / 2

        # 判定氢键供体：N or O 并且至少有一个隐含氢
        if atom_num in [7, 8] and num_hydrogens > 0 and lone_pairs > 0:
            donor_acceptors.append(atom_idx)
        elif atom_num in [7, 8] and num_hydrogens > 0:
            donors.append(atom_idx)
        elif lone_pairs > 0:
            acceptors.append(atom_idx)
        # 判定氢键受体：N, O, or F 并且没有隐含氢或仅部分隐含氢（排除氢键供体）
        # elif atom_num in [7, 8, 9] and (atom_num != 7 or num_hydrogens == 0):  # N, O, 或 F并且无隐含氢
        #     acceptors.append(atom_idx)
    return donors, acceptors, donor_acceptors


# 函数：识别疏水基团
def find_hydrophobic_groups(mol):
    hydrophobic_groups = []
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic() and atom.GetAtomicNum() == 6:  # Non-aromatic carbon atoms
            hydrophobic_groups.append(atom.GetIdx())
    return hydrophobic_groups


# 函数：识别芳香基团. pi-pi; 疏水
def find_aromatic_groups(mol):
    aromatic_groups = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_groups.append(atom.GetIdx())
    return aromatic_groups


# 函数：识别静电相互作用；盐桥
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


# 识别卤素。卤键
def find_halogens(mol):
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in ['Cl', 'Br', 'I']]



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
            score[idx] += 5
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

def run(base_path="E:/DATA/dgl_graphormer/geom_drugs", output_path='E:/DATA/dgl_graphormer/geom_drugs/processed'):
    drugs_file = os.path.join(base_path, "rdkit_folder/summary_drugs.json")
    sample_numbers = {1: 1, 2: 1, 3: 5, 4: 5, 5: 3, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}
    atom_types = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
    n_types = 9
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    mol_paths = []
    for smiles, sub_dic in drugs_summ.items():
        pickle_path = os.path.join(base_path, "rdkit_folder",
                                   sub_dic.get("pickle_path", ""))
        if os.path.isfile(pickle_path):
            mol_paths.append(pickle_path)

    train, val, test = [], [], []
    n = len(mol_paths)
    uuid_tr = 0
    uuid_val = 0
    uuid_te = 0
    t0 = time.time()
    n_mol = 0
    for i in tqdm(range(n)):
        if i % 1000 == 0:
            print(f"n :{i}, time: {time.time() - t0}")
        rand = random.random()
        mol_path = mol_paths[i]
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)

        # set the keys of the new dictionary to
        # be SMILES strings
        lowestenergy = float("inf")
        for conf in dic['conformers']:
            if conf['totalenergy'] < lowestenergy:
                lowestenergy = conf['totalenergy']
                mol = conf['rd_mol']

        mol = Chem.RemoveHs(mol)
        # ======================== num_atoms ========================
        num_nodes = mol.GetNumAtoms()
        if num_nodes > 40 or num_nodes < 5:
            continue

        # ======================== name ========================
        smi = Chem.MolToSmiles(mol)
        name = smi
        # ======================== positions ========================
        positions = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
        if torch.isnan(positions).any():
            continue
        # ======================== one hot of atom type ========================
        one_hot = []
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            if atom_symbol in atom_types:
                one_hot.append(atom_types[atom_symbol])
            else:
                continue
        if len(one_hot) < num_nodes:
            continue
        one_hot = torch.nn.functional.one_hot(torch.tensor(one_hot), n_types).type(torch.float)
        if torch.isnan(one_hot).any():
            continue
        # ======================== charges ========================
        charges = []
        for atom in mol.GetAtoms():
            atom_num = atom.GetAtomicNum()
            charges.append(atom_num)
        charges = torch.tensor(charges, dtype=torch.float)
        if torch.isnan(charges).any():
            continue
        # ======================== anchors ============================= #
        # 识别不同类型的相互作用位点
        h_bond_donors, h_bond_acceptors, donor_acceptors = find_h_bond_donors_and_acceptors(mol)
        hydrophobic_groups = find_hydrophobic_groups(mol)
        aromatic_groups = find_aromatic_groups(mol)
        positive_sites, negative_sites = find_electrostatic_interactions(mol)
        halogens = find_halogens(mol)
        # coordination_sites = find_coordination_sites(mol)

        # 作用力类型
        types_map = {}
        for idx in h_bond_donors:
            types_map[idx] = 1
        for idx in h_bond_acceptors:
            if idx not in types_map:
                types_map[idx] = 2
        for idx in hydrophobic_groups:
            if idx not in types_map:
                types_map[idx] = 3
        for idx in positive_sites:
            if idx not in types_map:
                types_map[idx] = 4
        for idx in negative_sites:
            if idx not in types_map:
                types_map[idx] = 5
        for idx in coordination_sites:
            if idx not in types_map:
                types_map[idx] = 6

        # 候选结合原子
        candidate_atoms = sort_bind_atom(mol, h_bond_donors, h_bond_acceptors, hydrophobic_groups, positive_sites,
                                         negative_sites, coordination_sites)
        if len(candidate_atoms) < 2:
            continue

        n_mol += 1

        # 采样需要分离的原子索引
        atom_indexes = []
        for k in range(1, min(len(sample_numbers)+1, len(candidate_atoms) + 1, num_nodes)):
            for t in range(sample_numbers[k]):
                atom_index = list(set(random.sample(candidate_atoms, k)))
                if atom_index not in atom_indexes:
                    atom_indexes.append(atom_index)

        for atom_index in atom_indexes:
            link_index = list(set([x for x in range(num_nodes)]) - set(atom_index))
            # nci
            nci = torch.zeros_like(charges)
            nci[:len(atom_index)] = torch.tensor([types_map[x] for x in atom_index])

            atom_index = torch.tensor(atom_index)
            anchors = torch.zeros(num_nodes)
            anchors[atom_index] = 1.
            if torch.isnan(anchors).any():
                continue

            # frag charges; frag one_hot; frag pos
            frag_charges = charges[atom_index]
            frag_one_hot = one_hot[atom_index]
            frag_pos = positions[atom_index]

            # link charges; link one_hot; link pos
            link_charges = charges[link_index]
            link_one_hot = one_hot[link_index]
            link_pos = positions[link_index]

            positions_ = torch.cat([frag_pos, link_pos])
            one_hot_ = torch.cat([frag_one_hot, link_one_hot])
            charges_ = torch.cat([frag_charges, link_charges])

            # ======================== fragment_mask ========================
            fragment_mask_ = torch.cat([torch.ones_like(frag_charges), torch.zeros_like(link_charges)])
            # ======================== linker_mask ========================
            linker_mask_ = torch.cat([torch.zeros_like(frag_charges), torch.ones_like(link_charges)])

            if rand < 0.0005:
                test.append({'uuid': uuid_te, 'name': name, 'positions': positions_, 'one_hot': one_hot_,
                             'charges': charges_, 'anchors': anchors, 'fragment_mask': fragment_mask_,
                             'linker_mask': linker_mask_, 'num_atoms': num_nodes, 'nci': nci})
                uuid_te += 1
            elif rand < 0.001:
                val.append({'uuid': uuid_val, 'name': name, 'positions': positions_, 'one_hot': one_hot_,
                            'charges': charges_, 'anchors': anchors, 'fragment_mask': fragment_mask_,
                            'linker_mask': linker_mask_, 'num_atoms': num_nodes, 'nci': nci})
                uuid_val += 1
            else:
                train.append({'uuid': uuid_tr, 'name': name, 'positions': positions_, 'one_hot': one_hot_,
                              'charges': charges_, 'anchors': anchors, 'fragment_mask': fragment_mask_,
                              'linker_mask': linker_mask_, 'num_atoms': num_nodes, 'nci': nci})
                uuid_tr += 1

    print(f"Total mols: {n_mol}")
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    torch.save(train, os.path.join(output_path, 'geom_multifrag_train.pt'))
    torch.save(val, os.path.join(output_path, 'geom_multifrag_val.pt'))
    torch.save(test, os.path.join(output_path, 'geom_multifrag_test.pt'))
    print("Finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geom-json', action='store', type=str, required=True)
    parser.add_argument('--output', action='store', type=str, required=True)
    args = parser.parse_args()

    run(
        base_path=args.geom_json,
        output_path=args.output,
    )


# pt = torch.load(r'E:\DATA\dgl_graphormer\geom_drugs\processed\geom_multifrag_test.pt')
'''
# target_example:
{'uuid': 0, 'name': 'C#CC(=O)N(c1ccc(OC)c(OC)c1)[C@@H](C(=O)NC1CCCC1)c1cccs1', 'positions': tensor([[ 1.9626,  0.7919,  3.2514],
        [ 2.8983,  0.0462,  2.5049],
        [ 2.5115, -0.4436,  1.3028],
        [ 1.2393, -0.3217,  0.7767],
        [ 0.9466, -0.8291, -0.4876],
        [ 1.9208, -1.4766, -1.2210],
        [ 3.1927, -1.6323, -0.6866],
        [ 3.5078, -1.1220,  0.5605],
        [ 4.7255, -1.2072,  1.1589],
        [ 5.7670, -1.8505,  0.4683],
        [-1.3337,  1.7587,  1.0097],
        [-1.7273,  1.0746,  0.0817],
        [-2.9311,  0.5043,  0.0096],
        [-3.8906,  0.5157,  1.0944],
        [-4.0100, -0.8854,  1.7113],
        [-5.0643, -1.5660,  0.8445],
        [-6.1083, -0.4664,  0.6555],
        [-5.2984,  0.8360,  0.5555],
        [ 1.1004,  1.8136, -2.3914],
        [ 0.2882,  1.7104, -1.3016],
        [ 0.7671,  2.7970, -0.0475],
        [ 2.0605,  3.3806, -1.0165],
        [ 2.1191,  2.7752, -2.2289],
        [-0.3150, -4.1160, -1.1672],
        [-0.5987, -2.9549, -1.2753],
        [-1.1454, -1.6314, -1.4382],
        [-2.2716, -1.4554, -1.8894],
        [-0.3502, -0.6221, -1.0229],
        [-0.8455,  0.7584, -1.1437]], device='cuda:0'), 'one_hot': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0'), 'charges': tensor([ 6.,  8.,  6.,  6.,  6.,  6.,  6.,  6.,  8.,  6.,  8.,  6.,  7.,  6.,
         6.,  6.,  6.,  6.,  6.,  6., 16.,  6.,  6.,  6.,  6.,  6.,  8.,  7.,
         6.], device='cuda:0'), 'anchors': tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0'), 'fragment_mask': tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], device='cuda:0'), 'linker_mask': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.], device='cuda:0'), 'num_atoms': 29}
'''