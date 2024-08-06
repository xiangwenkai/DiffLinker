import copy
import random
import numpy as np
import os
import pytorch_lightning as pl
import torch
import wandb
from src import metrics, utils, delinker
from src.const import LINKER_SIZE_DIST
from src.egnn import Dynamics, Pre_Dynamics, DynamicsWithPockets, Pre_DynamicsWithPockets
from src.edm import EDM, Pre_EDM, InpaintingEDM
from src.datasets import (
    ZincDataset, MOADDataset, create_templates_for_linker_generation, get_dataloader, collate, collate_pre
)
from src.linker_size import DistributionNodes
from src.molecule_builder import build_molecules
from src.visualizer import save_xyz_file, visualize_chain
from typing import Dict, List, Optional
from tqdm import tqdm
from src import const
from pdb import set_trace


def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")


class DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention, norm_constant,
        inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        normalize_factors, include_charges, model,
        data_path, train_data_prefix, val_data_prefix, batch_size, lr, torch_device, test_epochs, n_stability_samples,
        normalization=None, log_iterations=None, samples_dir=None, data_augmentation=False,
        center_of_mass='fragments', inpainting=False, anchors_context=True,
    ):
        super(DDPM, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.include_charges = include_charges
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.inpainting = inpainting
        self.loss_type = diffusion_loss_type

        self.n_dims = n_dims
        self.num_classes = in_node_nf - include_charges
        self.include_charges = include_charges
        self.anchors_context = anchors_context

        self.is_geom = ('geom' in self.train_data_prefix) or ('MOAD' in self.train_data_prefix)

        if type(activation) is str:
            activation = get_activation(activation)

        dynamics_class = DynamicsWithPockets if '.' in train_data_prefix else Dynamics
        dynamics = dynamics_class(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            context_node_nf=context_node_nf,
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            centering=inpainting,
        )
        edm_class = InpaintingEDM if inpainting else EDM
        self.edm = edm_class(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
        )
        self.linker_size_sampler = DistributionNodes(LINKER_SIZE_DIST)

    def setup(self, stage: Optional[str] = None):
        dataset_type = MOADDataset if '.' in self.train_data_prefix else ZincDataset
        if stage == 'fit':
            self.is_geom = ('geom' in self.train_data_prefix) or ('MOAD' in self.train_data_prefix)
            self.train_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.torch_device
            )
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        elif stage == 'val':
            self.is_geom = ('geom' in self.val_data_prefix) or ('MOAD' in self.val_data_prefix)
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        else:
            raise NotImplementedError

    def train_dataloader(self, collate_fn=collate_pre):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self, collate_fn=collate_pre):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=collate_pre):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn)

    def pre_process(self, batch):
        keys_to_remove = ['atom_indexes']
        if type(batch) == list:
            batch_copy = copy.deepcopy(batch)
            if 'atom_indexes' in batch_copy[0]:
                for x in batch_copy:
                    k = len(x['atom_indexes'])
                    k_sample = random.randint(0, k-1)
                    atom_index = x['atom_indexes'][k_sample]
                    n_frag = len(atom_index)
                    link_index = list(set([x for x in range(x['num_atoms'])]) - set(atom_index))
                    atom_index = torch.tensor(atom_index, dtype=int)
                    # nci_type = x['nci_types'][k_sample]

                    frag_charges = x['charges'][atom_index]
                    frag_one_hot = x['one_hot'][atom_index]
                    frag_pos = x['positions'][atom_index]

                    link_charges = x['charges'][link_index]
                    link_one_hot = x['one_hot'][link_index]
                    link_pos = x['positions'][link_index]

                    x['positions'] = torch.cat([frag_pos, link_pos])
                    x['one_hot'] = torch.cat([frag_one_hot, link_one_hot])
                    x['charges'] = torch.cat([frag_charges, link_charges])

                    # nci
                    # nci = torch.zeros_like(x['charges'])
                    # nci[:len(atom_index)] = torch.tensor(nci_type)
                    # x['nci'] = nci
                    # ======================== fragment_mask ========================
                    x['fragment_mask'] = torch.cat([
                        torch.ones_like(frag_charges),
                        torch.zeros_like(link_charges)
                    ])

                    x['anchors'] = x['fragment_mask'][:]

                    # ======================== linker_mask ========================
                    x['linker_mask'] = torch.cat([
                        torch.zeros_like(frag_charges),
                        torch.ones_like(link_charges)
                    ])

                    for key in keys_to_remove:
                        del x[key]
            out = {}
            for i, data in enumerate(batch_copy):
                for key, value in data.items():
                    out.setdefault(key, []).append(value)

            for key, value in out.items():
                if key in const.DATA_LIST_ATTRS:
                    continue
                if key in const.DATA_ATTRS_TO_PAD:
                    out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
                    continue
                raise Exception(f'Unknown batch key: {key}')

            atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
            out['atom_mask'] = atom_mask[:, :, None]

            batch_size, n_nodes = atom_mask.size()

            # In case of MOAD edge_mask is batch_idx
            if 'pocket_mask' in batch_copy[0].keys():
                batch_mask = torch.cat([
                    torch.ones(n_nodes, dtype=const.TORCH_INT) * i
                    for i in range(batch_size)
                ]).to(atom_mask.device)
                out['edge_mask'] = batch_mask
            else:
                edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
                edge_mask *= diag_mask
                out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

            for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                if key in out.keys():
                    out[key] = out[key][:, :, None]
            return out
        else:
            return batch


    def forward(self, data, training):
        data = self.pre_process(data)
        x = data['positions']
        h = data['one_hot']
        node_mask = data['atom_mask']
        edge_mask = data['edge_mask']
        anchors = data['anchors']
        fragment_mask = data['fragment_mask']
        linker_mask = data['linker_mask']
        # nci = data['nci']  # nci
        # padding = (0, 0, 0, x.shape[1] - nci.shape[1])  # nci
        # nci = torch.nn.functional.pad(nci, padding, "constant", value=0)  # nci
        # nci = torch.nn.functional.one_hot(nci.squeeze(2), num_classes=NCI_TYPE+1)  # nci
        # Anchors and fragments labels are used as context
        if self.anchors_context:
            context = torch.cat([anchors, fragment_mask], dim=-1)
            # context = torch.cat([anchors, fragment_mask, nci], dim=-1)  # nci
        else:
            context = fragment_mask
            # context = torch.cat([fragment_mask, nci], dim=-1)  # nci

        # Add information about pocket to the context
        if isinstance(self.train_dataset, MOADDataset):
            fragment_pocket_mask = fragment_mask
            fragment_only_mask = data['fragment_only_mask']
            pocket_only_mask = fragment_pocket_mask - fragment_only_mask
            if self.anchors_context:
                context = torch.cat([anchors, fragment_only_mask, pocket_only_mask], dim=-1)
                # context = torch.cat([anchors, fragment_only_mask, pocket_only_mask, nci], dim=-1)  # nci
            else:
                context = torch.cat([fragment_only_mask, pocket_only_mask], dim=-1)
                # context = torch.cat([anchors, fragment_only_mask, pocket_only_mask, nci], dim=-1)  # nci

        # Removing COM of fragment from the atom coordinates
        if self.inpainting:
            center_of_mass_mask = node_mask
        elif isinstance(self.train_dataset, MOADDataset) and self.center_of_mass == 'fragments':
            center_of_mass_mask = data['fragment_only_mask']
        elif self.center_of_mass == 'fragments':
            center_of_mass_mask = fragment_mask
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)
        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

        # Applying random rotation
        if training and self.data_augmentation:
            x = utils.random_rotation(x)

        return self.edm.forward(
            x=x,
            h=h,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context
        )

    def training_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=True)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)

        training_metrics = {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }
        if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
            for metric_name, metric in training_metrics.items():
                self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
                self.log(f'{metric_name}/train', metric, prog_bar=True)
        return training_metrics

    def validation_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        return {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

    def test_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        return {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

    def training_epoch_end(self, training_step_outputs):
        for metric in training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
        for metric in validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        if (self.current_epoch + 1) % self.test_epochs == 0:
            sampling_results = self.sample_and_analyze(self.val_dataloader())
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/val', metric_value, prog_bar=True)
                self.metrics.setdefault(f'{metric_name}/val', []).append(metric_value)

            # Logging the results corresponding to the best validation_and_connectivity
            best_metrics, best_epoch = self.compute_best_validation_metrics()
            self.log('best_epoch', int(best_epoch), prog_bar=True, batch_size=self.batch_size)
            for metric, value in best_metrics.items():
                self.log(f'best_{metric}', value, prog_bar=True, batch_size=self.batch_size)

    def test_epoch_end(self, test_step_outputs):
        for metric in test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(test_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/test', []).append(avg_metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)

        if (self.current_epoch + 1) % self.test_epochs == 0:
            sampling_results = self.sample_and_analyze(self.test_dataloader())
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/test', metric_value, prog_bar=True)
                self.metrics.setdefault(f'{metric_name}/test', []).append(metric_value)

    def generate_animation(self, chain_batch, node_mask, batch_i):
        batch_indices, mol_indices = utils.get_batch_idx_for_animation(self.batch_size, batch_i)
        for bi, mi in zip(batch_indices, mol_indices):
            try:
                chain = chain_batch[:, bi, :, :]
            except:
                continue
            name = f'mol_{mi}'
            chain_output = os.path.join(self.samples_dir, f'epoch_{self.current_epoch}', name)
            os.makedirs(chain_output, exist_ok=True)

            one_hot = chain[:, :, 3:-1] if self.include_charges else chain[:, :, 3:]
            positions = chain[:, :, :3]
            chain_node_mask = torch.cat([node_mask[bi].unsqueeze(0) for _ in range(self.FRAMES)], dim=0)
            names = [f'{name}_{j}' for j in range(self.FRAMES)]

            save_xyz_file(chain_output, one_hot, positions, chain_node_mask, names=names, is_geom=self.is_geom)
            visualize_chain(chain_output, wandb=wandb, mode=name, is_geom=self.is_geom)

    def sample_and_analyze(self, dataloader):
        pred_molecules = []
        true_molecules = []
        true_fragments = []

        for b, data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Sampling'):
            data = self.pre_process(data)
            atom_mask = data['atom_mask']
            fragment_mask = data['fragment_mask']

            # Save molecules without pockets
            if '.' in self.train_data_prefix:
                atom_mask = data['atom_mask'] - data['pocket_mask']
                fragment_mask = data['fragment_only_mask']

            true_molecules_batch = build_molecules(
                data['one_hot'],
                data['positions'],
                atom_mask,
                is_geom=self.is_geom,
            )
            true_fragments_batch = build_molecules(
                data['one_hot'],
                data['positions'],
                fragment_mask,
                is_geom=self.is_geom,
            )

            for sample_idx in tqdm(range(self.n_stability_samples)):
                try:
                    chain_batch, node_mask = self.sample_chain(data, keep_frames=self.FRAMES)
                except utils.FoundNaNException as e:
                    for idx in e.x_h_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [xh], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    for idx in e.only_x_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [x ], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    for idx in e.only_h_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [ h], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    continue

                # Get final molecules from chains – for computing metrics
                x, h = utils.split_features(
                    z=chain_batch[0],
                    n_dims=self.n_dims,
                    num_classes=self.num_classes,
                    include_charges=self.include_charges,
                )

                # Save molecules without pockets
                if '.' in self.train_data_prefix:
                    node_mask = node_mask - data['pocket_mask']

                one_hot = h['categorical']
                pred_molecules_batch = build_molecules(one_hot, x, node_mask, is_geom=self.is_geom)

                # Adding only results for valid ground truth molecules
                for pred_mol, true_mol, frag in zip(pred_molecules_batch, true_molecules_batch, true_fragments_batch):
                    if metrics.is_valid(true_mol):
                        pred_molecules.append(pred_mol)
                        true_molecules.append(true_mol)
                        true_fragments.append(frag)

                # Generate animation – will always do it for molecules with idx 0, 110 and 360
                if self.samples_dir is not None and sample_idx == 0:
                    self.generate_animation(chain_batch=chain_batch, node_mask=node_mask, batch_i=b)

        # Our own & DeLinker metrics
        our_metrics = metrics.compute_metrics(
            pred_molecules=pred_molecules,
            true_molecules=true_molecules
        )
        delinker_metrics = delinker.get_delinker_metrics(
            pred_molecules=pred_molecules,
            true_molecules=true_molecules,
            true_fragments=true_fragments
        )
        return {
            **our_metrics,
            **delinker_metrics
        }

    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        if sample_fn is None:
            linker_sizes = data['linker_mask'].sum(1).view(-1).int()
        else:
            linker_sizes = sample_fn(data)

        if self.inpainting:
            template_data = data
        else:
            template_data = create_templates_for_linker_generation(data, linker_sizes)

        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        anchors = template_data['anchors']
        fragment_mask = template_data['fragment_mask']
        linker_mask = template_data['linker_mask']
        # nci = template_data['nci']  # nci
        # padding = (0, 0, 0, x.shape[1] - nci.shape[1])  # nci
        # nci = torch.nn.functional.pad(nci, padding, "constant", value=0)  # nci
        # nci = torch.nn.functional.one_hot(nci.squeeze(2), num_classes=NCI_TYPE + 1)  # nci
        # Anchors and fragments labels are used as context
        if self.anchors_context:
            context = torch.cat([anchors, fragment_mask], dim=-1)
            # context = torch.cat([anchors, fragment_mask, nci], dim=-1)  # nci
        else:
            context = fragment_mask
            # context = torch.cat([fragment_mask, nci], dim=-1)  # nci

        # Add information about pocket to the context
        if '.' in self.train_data_prefix:
            fragment_pocket_mask = fragment_mask
            fragment_only_mask = template_data['fragment_only_mask']
            pocket_only_mask = fragment_pocket_mask - fragment_only_mask
            if self.anchors_context:
                context = torch.cat([anchors, fragment_only_mask, pocket_only_mask], dim=-1)
                # context = torch.cat([anchors, fragment_only_mask, pocket_only_mask, nci], dim=-1)  # nci
            else:
                context = torch.cat([fragment_only_mask, pocket_only_mask], dim=-1)
                # context = torch.cat([fragment_only_mask, pocket_only_mask, nci], dim=-1)  # nci

        # Removing COM of fragment from the atom coordinates
        if self.inpainting:
            center_of_mass_mask = node_mask
        elif isinstance(self.val_dataset, MOADDataset) and self.center_of_mass == 'fragments':
            center_of_mass_mask = template_data['fragment_only_mask']
        elif self.center_of_mass == 'fragments':
            center_of_mass_mask = fragment_mask
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)
        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)

        chain = self.edm.sample_chain(
            x=x,
            h=h,
            node_mask=node_mask,
            edge_mask=edge_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            context=context,
            keep_frames=keep_frames,
        )
        return chain, node_mask

    def configure_optimizers(self):
        return torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    def compute_best_validation_metrics(self):
        loss = self.metrics[f'validity_and_connectivity/val']
        best_epoch = np.argmax(loss)
        best_metrics = {
            metric_name: metric_values[best_epoch]
            for metric_name, metric_values in self.metrics.items()
            if metric_name.endswith('/val')
        }
        return best_metrics, best_epoch

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()


class Pre_DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention, norm_constant,
        inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        normalize_factors, include_charges, model, pre_model,
        data_path, train_data_prefix, val_data_prefix, batch_size, lr, torch_device, test_epochs, n_stability_samples,
        normalization=None, log_iterations=None, samples_dir=None, data_augmentation=False,
        center_of_mass='fragments', inpainting=False, anchors_context=True,
    ):
        super(Pre_DDPM, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.include_charges = include_charges
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.inpainting = inpainting
        self.loss_type = diffusion_loss_type

        self.n_dims = n_dims
        self.num_classes = in_node_nf - include_charges
        self.include_charges = include_charges
        self.anchors_context = anchors_context

        self.is_geom = ('geom' in self.train_data_prefix) or ('MOAD' in self.train_data_prefix)

        # pretrained model
        self.pre_edm = pre_model.edm

        if type(activation) is str:
            activation = get_activation(activation)

        dynamics_class = Pre_DynamicsWithPockets if '.' in train_data_prefix else Pre_Dynamics
        dynamics = dynamics_class(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            context_node_nf=context_node_nf,
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            pre_model=self.pre_edm,
            normalization=normalization,
            centering=inpainting,
        )
        edm_class = InpaintingEDM if inpainting else Pre_EDM
        self.edm = edm_class(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
        )
        self.linker_size_sampler = DistributionNodes(LINKER_SIZE_DIST)

    def setup(self, stage: Optional[str] = None):
        dataset_type = MOADDataset if '.' in self.train_data_prefix else ZincDataset
        if stage == 'fit':
            self.is_geom = ('geom' in self.train_data_prefix) or ('MOAD' in self.train_data_prefix)
            self.train_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.torch_device
            )
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        elif stage == 'val':
            self.is_geom = ('geom' in self.val_data_prefix) or ('MOAD' in self.val_data_prefix)
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        else:
            raise NotImplementedError

    def train_dataloader(self, collate_fn=collate_pre):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self, collate_fn=collate_pre):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=collate_pre):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn)

    def pre_process(self, batch):
        keys_to_remove = ['pocket_pos', 'pocket_one_hot', 'pocket_charges', 'atom_indexes',
                          'nci_types', 'mol_pos', 'mol_one_hot', 'mol_charges']
        if type(batch) == list:
            batch_copy = copy.deepcopy(batch)
            if 'atom_indexes' in batch_copy[0]:
                for x in batch_copy:
                    k = len(x['atom_indexes'])
                    k_sample = random.randint(0, k-1)
                    atom_index = x['atom_indexes'][k_sample]
                    n_frag = len(atom_index)
                    link_index = list(set([x for x in range(x['num_atoms'])]) - set(atom_index))
                    atom_index = torch.tensor(atom_index, dtype=int)
                    # nci_type = x['nci_types'][k_sample]

                    frag_charges = x['mol_charges'][atom_index]
                    frag_one_hot = x['mol_one_hot'][atom_index]
                    frag_pos = x['mol_pos'][atom_index]

                    link_charges = x['mol_charges'][link_index]
                    link_one_hot = x['mol_one_hot'][link_index]
                    link_pos = x['mol_pos'][link_index]

                    x['positions'] = torch.cat([frag_pos, x['pocket_pos'], link_pos])
                    x['one_hot'] = torch.cat([frag_one_hot, x['pocket_one_hot'], link_one_hot])
                    x['charges'] = torch.cat([frag_charges, x['pocket_charges'], link_charges])


                    # nci
                    # nci = torch.zeros_like(x['charges'])
                    # nci[:len(atom_index)] = torch.tensor(nci_type)
                    # x['nci'] = nci
                    # ======================== fragment_mask ========================
                    x['fragment_only_mask'] = torch.cat([
                        torch.ones_like(frag_charges),
                        torch.zeros_like(x['pocket_charges']),
                        torch.zeros_like(link_charges)
                    ])

                    x['anchors'] = x['fragment_only_mask'][:]

                    x['pocket_mask'] = torch.cat([
                        torch.zeros_like(frag_charges),
                        torch.ones_like(x['pocket_charges']),
                        torch.zeros_like(link_charges)
                    ])
                    x['fragment_mask'] = torch.cat([
                        torch.ones_like(frag_charges),
                        torch.ones_like(x['pocket_charges']),
                        torch.zeros_like(link_charges)
                    ])
                    # ======================== linker_mask ========================
                    x['linker_mask'] = torch.cat([
                        torch.zeros_like(frag_charges),
                        torch.zeros_like(x['pocket_charges']),
                        torch.ones_like(link_charges)
                    ])

                    x['positions_pre'] = torch.cat([frag_pos, link_pos])
                    x['one_hot_pre'] = torch.cat([frag_one_hot, link_one_hot])
                    x['charges_pre'] = torch.cat([frag_charges, link_charges])
                    x['anchors_pre'] = x['anchors'][:x['num_atoms']]
                    x['fragment_mask_pre'] = torch.cat([torch.ones_like(frag_charges), torch.zeros_like(link_charges)])
                    x['linker_mask_pre'] = torch.cat([torch.zeros_like(frag_charges), torch.ones_like(link_charges)])
                    # x['nci_pre'] = x['nci'][:x['num_atoms']]

                    # mol index
                    n_poc = len(x['pocket_charges'])
                    x['mol_index'] = [n_frag, n_frag + n_poc, n_poc + x['num_atoms']]

                    for key in keys_to_remove:
                        del x[key]
            elif 'positions_pre' not in batch_copy[0]:
                for x in batch_copy:
                    n_atoms = x['num_atoms']
                    n_frag = int(sum(x['fragment_only_mask']).item())
                    n_poc = int(sum(x['pocket_mask']).item())
                    frag_pos = x['positions'][:n_frag]
                    link_pos = x['positions'][n_frag + n_poc: n_atoms]
                    frag_one_hot = x['one_hot'][:n_frag]
                    link_one_hot = x['one_hot'][n_frag + n_poc: n_atoms]
                    frag_charges = x['charges'][:n_frag]
                    link_charges = x['charges'][n_frag + n_poc: n_atoms]
                    x['positions_pre'] = torch.cat([frag_pos, link_pos])
                    x['one_hot_pre'] = torch.cat([frag_one_hot, link_one_hot])
                    x['charges_pre'] = torch.cat([frag_charges, link_charges])
                    x['anchors_pre'] = x['anchors'][:n_atoms - n_poc]
                    x['fragment_mask_pre'] = torch.cat([torch.ones_like(frag_charges), torch.zeros_like(link_charges)])
                    x['linker_mask_pre'] = torch.cat([torch.zeros_like(frag_charges), torch.ones_like(link_charges)])
                    x['mol_index'] = [n_frag, n_frag + n_poc, n_atoms]
            out = {}
            for i, data in enumerate(batch_copy):
                for key, value in data.items():
                    out.setdefault(key, []).append(value)

            for key, value in out.items():
                if key in const.DATA_LIST_ATTRS:
                    continue
                if key in const.DATA_ATTRS_TO_PAD:
                    out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
                    continue
                raise Exception(f'Unknown batch key: {key}')

            atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
            out['atom_mask'] = atom_mask[:, :, None]

            atom_mask_pre = (out['fragment_mask_pre'].bool() | out['linker_mask_pre'].bool()).to(const.TORCH_INT)
            out['atom_mask_pre'] = atom_mask_pre[:, :, None]

            batch_size, n_nodes = atom_mask.size()
            _, n_nodes_pre = atom_mask_pre.size()

            # In case of MOAD edge_mask is batch_idx
            if 'pocket_mask' in batch_copy[0].keys():
                batch_mask = torch.cat([
                    torch.ones(n_nodes, dtype=const.TORCH_INT) * i
                    for i in range(batch_size)
                ]).to(atom_mask.device)
                out['edge_mask'] = batch_mask

                edge_mask_pre = atom_mask_pre[:, None, :] * atom_mask_pre[:, :, None]
                diag_mask_pre = ~torch.eye(edge_mask_pre.size(1), dtype=const.TORCH_INT,
                                           device=atom_mask_pre.device).unsqueeze(0)
                edge_mask_pre *= diag_mask_pre
                out['edge_mask_pre'] = edge_mask_pre.view(batch_size * n_nodes_pre * n_nodes_pre, 1)
            else:
                edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
                edge_mask *= diag_mask
                out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

            for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                if key in out.keys():
                    out[key] = out[key][:, :, None]
            return out
        else:
            return batch


    def forward(self, data, training):
        data = self.pre_process(data)
        x = data['positions']
        h = data['one_hot']
        node_mask = data['atom_mask']
        edge_mask = data['edge_mask']
        anchors = data['anchors']
        fragment_mask = data['fragment_mask']
        linker_mask = data['linker_mask']
        # nci = data['nci']  # nci
        x_pre = data['positions_pre']
        h_pre = data['one_hot_pre']
        node_mask_pre = data['atom_mask_pre']
        edge_mask_pre = data['edge_mask_pre']
        anchors_pre = data['anchors_pre']
        fragment_mask_pre = data['fragment_mask_pre']
        linker_mask_pre = data['linker_mask_pre']
        # nci_pre = data['nci_pre']  # nci

        # Anchors and fragments labels are used as context
        if self.anchors_context:
            context = torch.cat([anchors, fragment_mask], dim=-1)
            context_pre = torch.cat([anchors_pre, fragment_mask_pre], dim=-1)
            # context = torch.cat([anchors, fragment_mask, nci], dim=-1)  # nci
            # context_pre = torch.cat([anchors_pre, fragment_mask_pre, nci_pre], dim=-1)
        else:
            context = fragment_mask
            context_pre = torch.cat([anchors_pre, fragment_mask_pre], dim=-1)
            # context = torch.cat([fragment_mask, nci], dim=-1)  # nci
            # context_pre = torch.cat([anchors_pre, fragment_mask_pre, nci_pre], dim=-1)

        # Add information about pocket to the context
        if isinstance(self.train_dataset, MOADDataset):
            fragment_pocket_mask = fragment_mask
            fragment_only_mask = data['fragment_only_mask']
            pocket_only_mask = fragment_pocket_mask - fragment_only_mask
            if self.anchors_context:
                context = torch.cat([anchors, fragment_only_mask, pocket_only_mask], dim=-1)
                # context = torch.cat([anchors, fragment_only_mask, pocket_only_mask, nci], dim=-1)  # nci
            else:
                context = torch.cat([fragment_only_mask, pocket_only_mask], dim=-1)
                # context = torch.cat([anchors, fragment_only_mask, pocket_only_mask, nci], dim=-1)  # nci

        # Removing COM of fragment from the atom coordinates
        if self.inpainting:
            center_of_mass_mask = node_mask
        elif isinstance(self.train_dataset, MOADDataset) and self.center_of_mass == 'fragments':
            center_of_mass_mask = data['fragment_only_mask']
        elif self.center_of_mass == 'fragments':
            center_of_mass_mask = fragment_mask
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)
        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

        center_of_mass_mask_pre = anchors_pre
        x_pre = utils.remove_partial_mean_with_mask(x_pre, node_mask_pre, center_of_mass_mask_pre)
        utils.assert_partial_mean_zero_with_mask(x_pre, node_mask_pre, center_of_mass_mask_pre)

        pre_info = {'x': x_pre, 'h': h_pre, 'node_mask': node_mask_pre,
                    'fragment_mask': fragment_mask_pre, 'linker_mask': linker_mask_pre,
                    'edge_mask': edge_mask_pre, 'context': context_pre, 'mol_index': data['mol_index']}

        # Applying random rotation
        if training and self.data_augmentation:
            x = utils.random_rotation(x)

        return self.edm.forward(
            x=x,
            h=h,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
            pre_info=pre_info,
        )

    def training_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=True)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)

        training_metrics = {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }
        if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
            for metric_name, metric in training_metrics.items():
                self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
                self.log(f'{metric_name}/train', metric, prog_bar=True)
        return training_metrics

    def validation_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        return {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

    def test_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        return {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

    def training_epoch_end(self, training_step_outputs):
        for metric in training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
        for metric in validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        if (self.current_epoch + 1) % self.test_epochs == 0:
            sampling_results = self.sample_and_analyze(self.val_dataloader())
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/val', metric_value, prog_bar=True)
                self.metrics.setdefault(f'{metric_name}/val', []).append(metric_value)

            # Logging the results corresponding to the best validation_and_connectivity
            best_metrics, best_epoch = self.compute_best_validation_metrics()
            self.log('best_epoch', int(best_epoch), prog_bar=True, batch_size=self.batch_size)
            for metric, value in best_metrics.items():
                self.log(f'best_{metric}', value, prog_bar=True, batch_size=self.batch_size)

    def test_epoch_end(self, test_step_outputs):
        for metric in test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(test_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/test', []).append(avg_metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)

        if (self.current_epoch + 1) % self.test_epochs == 0:
            sampling_results = self.sample_and_analyze(self.test_dataloader())
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/test', metric_value, prog_bar=True)
                self.metrics.setdefault(f'{metric_name}/test', []).append(metric_value)

    def generate_animation(self, chain_batch, node_mask, batch_i):
        batch_indices, mol_indices = utils.get_batch_idx_for_animation(self.batch_size, batch_i)
        for bi, mi in zip(batch_indices, mol_indices):
            try:
                chain = chain_batch[:, bi, :, :]
            except:
                continue
            name = f'mol_{mi}'
            chain_output = os.path.join(self.samples_dir, f'epoch_{self.current_epoch}', name)
            os.makedirs(chain_output, exist_ok=True)

            one_hot = chain[:, :, 3:-1] if self.include_charges else chain[:, :, 3:]
            positions = chain[:, :, :3]
            chain_node_mask = torch.cat([node_mask[bi].unsqueeze(0) for _ in range(self.FRAMES)], dim=0)
            names = [f'{name}_{j}' for j in range(self.FRAMES)]

            save_xyz_file(chain_output, one_hot, positions, chain_node_mask, names=names, is_geom=self.is_geom)
            visualize_chain(chain_output, wandb=wandb, mode=name, is_geom=self.is_geom)

    def sample_and_analyze(self, dataloader):
        pred_molecules = []
        true_molecules = []
        true_fragments = []

        for b, data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Sampling'):
            data = self.pre_process(data)
            atom_mask = data['atom_mask']
            fragment_mask = data['fragment_mask']

            # Save molecules without pockets
            if '.' in self.train_data_prefix:
                atom_mask = data['atom_mask'] - data['pocket_mask']
                fragment_mask = data['fragment_only_mask']

            true_molecules_batch = build_molecules(
                data['one_hot'],
                data['positions'],
                atom_mask,
                is_geom=self.is_geom,
            )
            true_fragments_batch = build_molecules(
                data['one_hot'],
                data['positions'],
                fragment_mask,
                is_geom=self.is_geom,
            )

            for sample_idx in tqdm(range(self.n_stability_samples)):
                try:
                    chain_batch, node_mask = self.sample_chain(data, keep_frames=self.FRAMES)
                except utils.FoundNaNException as e:
                    for idx in e.x_h_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [xh], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    for idx in e.only_x_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [x ], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    for idx in e.only_h_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [ h], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    continue

                # Get final molecules from chains – for computing metrics
                x, h = utils.split_features(
                    z=chain_batch[0],
                    n_dims=self.n_dims,
                    num_classes=self.num_classes,
                    include_charges=self.include_charges,
                )

                # Save molecules without pockets
                if '.' in self.train_data_prefix:
                    node_mask = node_mask - data['pocket_mask']

                one_hot = h['categorical']
                pred_molecules_batch = build_molecules(one_hot, x, node_mask, is_geom=self.is_geom)

                # Adding only results for valid ground truth molecules
                for pred_mol, true_mol, frag in zip(pred_molecules_batch, true_molecules_batch, true_fragments_batch):
                    if metrics.is_valid(true_mol):
                        pred_molecules.append(pred_mol)
                        true_molecules.append(true_mol)
                        true_fragments.append(frag)

                # Generate animation – will always do it for molecules with idx 0, 110 and 360
                if self.samples_dir is not None and sample_idx == 0:
                    self.generate_animation(chain_batch=chain_batch, node_mask=node_mask, batch_i=b)

        # Our own & DeLinker metrics
        our_metrics = metrics.compute_metrics(
            pred_molecules=pred_molecules,
            true_molecules=true_molecules
        )
        delinker_metrics = delinker.get_delinker_metrics(
            pred_molecules=pred_molecules,
            true_molecules=true_molecules,
            true_fragments=true_fragments
        )
        return {
            **our_metrics,
            **delinker_metrics
        }

    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        if sample_fn is None:
            linker_sizes = data['linker_mask'].sum(1).view(-1).int()
        else:
            linker_sizes = sample_fn(data)

        if self.inpainting:
            template_data = data
        else:
            template_data = create_templates_for_linker_generation(data, linker_sizes)

        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        anchors = template_data['anchors']
        fragment_mask = template_data['fragment_mask']
        linker_mask = template_data['linker_mask']
        # nci = template_data['nci']  # nci

        x_pre = data['positions_pre']
        h_pre = data['one_hot_pre']
        node_mask_pre = data['atom_mask_pre']
        edge_mask_pre = data['edge_mask_pre']
        anchors_pre = data['anchors_pre']
        fragment_mask_pre = data['fragment_mask_pre']
        linker_mask_pre = data['linker_mask_pre']
        # nci_pre = data['nci_pre']
        # context_pre = torch.cat([anchors_pre, fragment_mask_pre, nci_pre], dim=-1)
        context_pre = torch.cat([anchors_pre, fragment_mask_pre], dim=-1)

        # Anchors and fragments labels are used as context
        if self.anchors_context:
            context = torch.cat([anchors, fragment_mask], dim=-1)
            # context = torch.cat([anchors, fragment_mask, nci], dim=-1)  # nci
        else:
            context = fragment_mask
            # context = torch.cat([fragment_mask, nci], dim=-1)  # nci

        # Add information about pocket to the context
        if '.' in self.train_data_prefix:
            fragment_pocket_mask = fragment_mask
            fragment_only_mask = template_data['fragment_only_mask']
            pocket_only_mask = fragment_pocket_mask - fragment_only_mask
            if self.anchors_context:
                context = torch.cat([anchors, fragment_only_mask, pocket_only_mask], dim=-1)
                # context = torch.cat([anchors, fragment_only_mask, pocket_only_mask, nci], dim=-1)  # nci
            else:
                context = torch.cat([fragment_only_mask, pocket_only_mask], dim=-1)
                # context = torch.cat([fragment_only_mask, pocket_only_mask, nci], dim=-1)  # nci

        # Removing COM of fragment from the atom coordinates
        if self.inpainting:
            center_of_mass_mask = node_mask
        elif isinstance(self.val_dataset, MOADDataset) and self.center_of_mass == 'fragments':
            center_of_mass_mask = template_data['fragment_only_mask']
        elif self.center_of_mass == 'fragments':
            center_of_mass_mask = fragment_mask
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)
        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)

        center_of_mass_mask_pre = anchors_pre
        x_pre = utils.remove_partial_mean_with_mask(x_pre, node_mask_pre, center_of_mass_mask_pre)
        utils.assert_partial_mean_zero_with_mask(x_pre, node_mask_pre, center_of_mass_mask_pre)
        pre_info = {'x': x_pre, 'h': h_pre, 'node_mask': node_mask_pre,
                    'fragment_mask': fragment_mask_pre, 'linker_mask': linker_mask_pre,
                    'edge_mask': edge_mask_pre, 'context': context_pre, 'mol_index': data['mol_index']}

        chain = self.edm.sample_chain(
            x=x,
            h=h,
            node_mask=node_mask,
            edge_mask=edge_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            context=context,
            keep_frames=keep_frames,
            pre_info=pre_info,
        )
        return chain, node_mask

    def configure_optimizers(self):
        return torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    def compute_best_validation_metrics(self):
        loss = self.metrics[f'validity_and_connectivity/val']
        best_epoch = np.argmax(loss)
        best_metrics = {
            metric_name: metric_values[best_epoch]
            for metric_name, metric_values in self.metrics.items()
            if metric_name.endswith('/val')
        }
        return best_metrics, best_epoch

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()