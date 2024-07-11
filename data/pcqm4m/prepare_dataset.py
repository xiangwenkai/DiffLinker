from rdkit import Chem

suppl = Chem.SDMolSupplier('E:/DATA/pcqm4m/pcqm4m-v2-train.sdf')
for idx, mol in enumerate(suppl):
    print(f'{idx}-th rdkit mol obj: {mol}')






