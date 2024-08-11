import copy
import math
import numpy as np
import torch
import torch.nn as nn
from src.const import ATOM_TYPE
from src import utils
from pdb import set_trace


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method, activation,
                 edges_in_d=0, nodes_att_dim=0, attention=False, normalization=None):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation)

        if normalization is None:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf)
            )
        elif normalization == 'batch_norm':
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.BatchNorm1d(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.BatchNorm1d(output_nf),
            )
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, activation=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, linker_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if linker_mask is not None:
            agg = agg * linker_mask

        coord = coord + agg
        return coord

    def forward(
            self, h, coord, edge_index, coord_diff, edge_attr=None, linker_mask=None, node_mask=None, edge_mask=None
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask, linker_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', activation=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              activation=activation, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, activation=activation, tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        if torch.cuda.is_available():
            self.to(self.device)
        else:
            self.to('cpu')

    def forward(self, h, x, edge_index, node_mask=None, linker_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](
            h, x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
            linker_mask=linker_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', activation=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               activation=activation, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        if torch.cuda.is_available():
            self.to(self.device)
        else:
            self.to('cpu')

    def forward(self, h, x, edge_index, node_mask=None, linker_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)

        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                edge_attr=distances
            )

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 activation=nn.SiLU(), n_layers=4, attention=False, normalization_factor=1,
                 out_node_nf=None, normalization=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, activation=activation,
                attention=attention, normalization=normalization))

        if torch.cuda.is_available():
            self.to(self.device)
        else:
            self.to('cpu')

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class Dynamics(nn.Module):
    def __init__(
            self, n_dims, in_node_nf, context_node_nf, hidden_nf=64, device='cpu', activation=nn.SiLU(),
            n_layers=4, attention=False, condition_time=True, tanh=False, norm_constant=0, inv_sublayers=2,
            sin_embedding=False, normalization_factor=100, aggregation_method='sum', model='egnn_dynamics',
            normalization=None, centering=False, graph_type='FC'
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.centering = centering
        self.graph_type = graph_type

        in_node_nf = in_node_nf + context_node_nf + condition_time
        if self.model == 'egnn_dynamics':
            self.dynamics = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf, device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
        elif self.model == 'gnn_dynamics':
            self.dynamics = GNN(
                in_node_nf=in_node_nf+3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=in_node_nf+3,
                device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                normalization=normalization,
            )
        else:
            raise NotImplementedError

        self.edge_cache = {}

    def forward(self, t, xh, node_mask, linker_mask, edge_mask, context):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        """

        assert self.graph_type == 'FC'

        bs, n_nodes = xh.shape[0], xh.shape[1]
        edges = self.get_edges(n_nodes, bs)  # (2, B*N)
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if linker_mask is not None:
            linker_mask = linker_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(h,x,edges,node_mask=node_mask,linker_mask=linker_mask,edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        if self.centering:
            vel = utils.remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, h_final], dim=2)

    def get_edges(self, n_nodes, batch_size):
        if n_nodes in self.edge_cache:
            edges_dic_b = self.edge_cache[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self.edge_cache[n_nodes] = {}
            return self.get_edges(n_nodes, batch_size)


class DynamicsWithPockets(Dynamics):
    def forward(self, t, xh, node_mask, linker_mask, edge_mask, context):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        """

        bs, n_nodes = xh.shape[0], xh.shape[1]
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if linker_mask is not None:
            linker_mask = linker_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        fragment_only_mask = context[..., -2].view(bs * n_nodes, 1)  # (B*N, 1)
        pocket_only_mask = context[..., -1].view(bs * n_nodes, 1)  # (B*N, 1)
        assert torch.all(fragment_only_mask.bool() | pocket_only_mask.bool() | linker_mask.bool() == node_mask.bool())

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)

        assert self.graph_type in ['4A', 'FC-4A', 'FC-10A-4A']
        if self.graph_type == '4A' or self.graph_type is None:
            edges = self.get_dist_edges_4A(x, node_mask, edge_mask)
        else:
            edges = self.get_dist_edges(x, node_mask, edge_mask, linker_mask, fragment_only_mask, pocket_only_mask)

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=None
            )
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        if self.centering:
            vel = utils.remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, h_final], dim=2)

    @staticmethod
    def get_dist_edges_4A(x, node_mask, batch_mask):
        node_mask = node_mask.squeeze().bool()
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        dists_adj = (torch.cdist(x, x) <= 4)
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        adj = batch_adj & nodes_adj & dists_adj & rm_self_loops
        edges = torch.stack(torch.where(adj))
        return edges

    def get_dist_edges(self, x, node_mask, batch_mask, linker_mask, fragment_only_mask, pocket_only_mask):
        node_mask = node_mask.squeeze().bool()
        linker_mask = linker_mask.squeeze().bool() & node_mask
        fragment_only_mask = fragment_only_mask.squeeze().bool() & node_mask
        pocket_only_mask = pocket_only_mask.squeeze().bool() & node_mask
        ligand_mask = linker_mask | fragment_only_mask

        # General constrains:
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        constraints = batch_adj & nodes_adj & rm_self_loops

        # Ligand atoms – fully-connected graph
        ligand_adj = (ligand_mask[:, None] & ligand_mask[None, :])
        ligand_interactions = ligand_adj & constraints

        # Pocket atoms - within 4A
        pocket_adj = (pocket_only_mask[:, None] & pocket_only_mask[None, :])
        pocket_dists_adj = (torch.cdist(x, x) <= 4)
        pocket_interactions = pocket_adj & pocket_dists_adj & constraints

        # Pocket-ligand atoms - within 10A
        pocket_ligand_cutoff = 4 if self.graph_type == 'FC-4A' else 10
        pocket_ligand_adj = (ligand_mask[:, None] & pocket_only_mask[None, :])
        pocket_ligand_adj = pocket_ligand_adj | (pocket_only_mask[:, None] & ligand_mask[None, :])
        pocket_ligand_dists_adj = (torch.cdist(x, x) <= pocket_ligand_cutoff)
        pocket_ligand_interactions = pocket_ligand_adj & pocket_ligand_dists_adj & constraints

        adj = ligand_interactions | pocket_interactions | pocket_ligand_interactions
        edges = torch.stack(torch.where(adj))
        return edges


class Pre_Dynamics(nn.Module):
    def __init__(
            self, n_dims, in_node_nf, context_node_nf,
            hidden_nf=64, device='cpu', activation=nn.SiLU(),
            n_layers=4, attention=False, condition_time=True, tanh=False, norm_constant=0, inv_sublayers=2,
            sin_embedding=False, normalization_factor=100, aggregation_method='sum', model='egnn_dynamics',
            pre_model=None, normalization=None, centering=False, graph_type='FC',
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.centering = centering
        self.graph_type = graph_type
        self.context_node_nf_pre = 2

        self.pre_edm = pre_model
        self.gamma = self.pre_edm.gamma
        for name, params in self.pre_edm.named_parameters():
            params.requires_grad = False
        # for name, params in self.pre_edm.named_parameters():
        #     if "embedding_out" in name:
        #         params.requires_grad = True
        #         print(f"{name} trained")
        #     else:
        #         params.requires_grad = False

        in_node_nf = in_node_nf + context_node_nf + condition_time
        if self.model == 'egnn_dynamics':
            self.dynamics = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf, device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
        elif self.model == 'gnn_dynamics':
            self.dynamics = GNN(
                in_node_nf=in_node_nf+3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=in_node_nf+3,
                device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                normalization=normalization,
            )
        else:
            raise NotImplementedError

        self.edge_cache = {}
        # self.w1 = nn.Parameter(0.5 * torch.ones([]))
        # self.w2 = nn.Parameter(0.5 * torch.ones([]))
        self.w1 = 0.5
        # self.w2 = 0.5

    def forward(self, t, xh, node_mask, linker_mask, edge_mask, context, pre_info=None):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        """

        assert self.graph_type == 'FC'

        bs, n_nodes = xh.shape[0], xh.shape[1]
        # edges = self.get_edges(n_nodes, bs)  # (2, B*N)
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if linker_mask is not None:
            linker_mask = linker_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        fragment_only_mask = context[..., -2].view(bs * n_nodes, 1)  # (B*N, 1)
        pocket_only_mask = context[..., -1].view(bs * n_nodes, 1)  # (B*N, 1)
        assert torch.all(fragment_only_mask.bool() | pocket_only_mask.bool() | linker_mask.bool() == node_mask.bool())

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)

        assert self.graph_type in ['4A', 'FC-4A', 'FC-10A-4A']
        if self.graph_type == '4A' or self.graph_type is None:
            edges = self.get_dist_edges_4A(x, node_mask, edge_mask)
        else:
            edges = self.get_dist_edges(x, node_mask, edge_mask, linker_mask, fragment_only_mask, pocket_only_mask)

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask
            )
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        if self.centering:
            vel = utils.remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, h_final], dim=2)

    @staticmethod
    def get_dist_edges_4A(x, node_mask, batch_mask):
        node_mask = node_mask.squeeze().bool()
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        dists_adj = (torch.cdist(x, x) <= 4)
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        adj = batch_adj & nodes_adj & dists_adj & rm_self_loops
        edges = torch.stack(torch.where(adj))
        return edges

    def get_dist_edges(self, x, node_mask, batch_mask, linker_mask, fragment_only_mask, pocket_only_mask):
        node_mask = node_mask.squeeze().bool()
        linker_mask = linker_mask.squeeze().bool() & node_mask
        fragment_only_mask = fragment_only_mask.squeeze().bool() & node_mask
        pocket_only_mask = pocket_only_mask.squeeze().bool() & node_mask
        ligand_mask = linker_mask | fragment_only_mask

        # General constrains:
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        constraints = batch_adj & nodes_adj & rm_self_loops

        # Ligand atoms – fully-connected graph
        ligand_adj = (ligand_mask[:, None] & ligand_mask[None, :])
        ligand_interactions = ligand_adj & constraints

        # Pocket atoms - within 4A
        pocket_adj = (pocket_only_mask[:, None] & pocket_only_mask[None, :])
        pocket_dists_adj = (torch.cdist(x, x) <= 4)
        pocket_interactions = pocket_adj & pocket_dists_adj & constraints

        # Pocket-ligand atoms - within 10A
        pocket_ligand_cutoff = 4 if self.graph_type == 'FC-4A' else 10
        pocket_ligand_adj = (ligand_mask[:, None] & pocket_only_mask[None, :])
        pocket_ligand_adj = pocket_ligand_adj | (pocket_only_mask[:, None] & ligand_mask[None, :])
        pocket_ligand_dists_adj = (torch.cdist(x, x) <= pocket_ligand_cutoff)
        pocket_ligand_interactions = pocket_ligand_adj & pocket_ligand_dists_adj & constraints

        adj = ligand_interactions | pocket_interactions | pocket_ligand_interactions
        edges = torch.stack(torch.where(adj))
        return edges


class Pre_DynamicsWithPockets(Pre_Dynamics):
    def forward(self, t, xh, node_mask, linker_mask, edge_mask, context, pre_info=None):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        """
        xh_update = xh.clone()

        if pre_info is not None:
            '''
            # pretraining information
            bs_pre, n_nodes_pre = pre_info['xh'].shape[0], pre_info['xh'].shape[1]
            edges_pre = self.get_edges(n_nodes_pre, bs_pre)  # (2, B*N)
            node_mask_pre = pre_info['node_mask'].view(bs_pre * n_nodes_pre, 1)  # (B*N, 1)

            linker_mask_pre = pre_info['linker_mask'].view(bs_pre * n_nodes_pre, 1)  # (B*N, 1)

            # 先加一步噪声，再还原
            gamma_t = self.inflate_batch_array(self.gamma(t), pre_info['x'])
            alpha_t = self.alpha(gamma_t, pre_info['x'])
            sigma_t = self.sigma(gamma_t, pre_info['x'])
            eps_t = self.sample_combined_position_feature_noise(n_samples=pre_info['x'].size(0), n_nodes=pre_info['x'].size(1),
                                                                mask=pre_info['linker_mask'])
            pre_info['xh'] = pre_info['xh'] * pre_info['fragment_mask'] + (alpha_t * pre_info['xh'] + sigma_t * eps_t) * pre_info['linker_mask']

            # Reshaping node features & adding time feature
            xh_pre = pre_info['xh'].view(bs_pre * n_nodes_pre, -1).clone() * node_mask_pre  # (B*N, D)
            x_pre = xh_pre[:, :self.n_dims].clone()  # (B*N, 3)
            h_pre = xh_pre[:, self.n_dims:].clone()  # (B*N, nf)
            t_pre = torch.clamp(t-1, min=0., max=10000.)
            if self.condition_time:
                if np.prod(t_pre.size()) == 1:
                    # t is the same for all elements in batch.
                    h_time_pre = torch.empty_like(h_pre[:, 0:1]).fill_(t_pre.item())
                else:
                    # t is different over the batch dimension.
                    h_time_pre = t_pre.view(bs_pre, 1).repeat(1, n_nodes_pre)
                    h_time_pre = h_time_pre.view(bs_pre * n_nodes_pre, 1)
                h_pre = torch.cat([h_pre, h_time_pre], dim=1)  # (B*N, nf+1)
            if pre_info['context'] is not None:
                context_pre = pre_info['context'].view(bs_pre * n_nodes_pre, self.context_node_nf_pre)
                h_pre = torch.cat([h_pre, context_pre], dim=1)

            # with torch.no_grad():
            h_final_pre, x_final_pre = self.pre_edm.dynamics.dynamics(h_pre, x_pre, edges_pre,
                                                                      node_mask=node_mask_pre,
                                                                      linker_mask=linker_mask_pre,
                                                                      edge_mask=pre_info['edge_mask'])
            # print(f"x_pre coords: {x_final_pre[pre_info['mol_index'][0][0]]}")

            # =========================processing===============================
            vel_pre = (x_final_pre - x_pre) * node_mask_pre
            if context is not None:
                h_final_pre = h_final_pre[:, :-self.context_node_nf_pre]
            if self.condition_time:
                h_final_pre = h_final_pre[:, :-1]
            vel_pre = vel_pre.view(bs_pre, n_nodes_pre, -1)  # (B, N, 3)
            h_final_pre = h_final_pre.view(bs_pre, n_nodes_pre, -1)  # (B, N, D)
            node_mask_pre = node_mask_pre.view(bs_pre, n_nodes_pre, 1)  # (B, N, 1)
            if torch.any(torch.isnan(vel_pre)) or torch.any(torch.isnan(h_final_pre)):
                raise utils.FoundNaNException(vel_pre, h_final_pre)
            if self.centering:
                vel_pre = utils.remove_mean_with_mask(vel_pre, node_mask_pre)
            eps_hat_pre = torch.cat([vel_pre, h_final_pre], dim=2)
            eps_hat_pre = eps_hat_pre * pre_info['linker_mask']

            t_is_zero = (t_pre == 0.).float().unsqueeze(dim=-1)
            t_is_not_zero = 1. - t_is_zero

            zero_num = sum(t_is_zero).item()
            if zero_num == 0:
                mu_pre = pre_info['xh'] / pre_info['alpha_t_given_s'] - (
                        pre_info['sigma2_t_given_s'] / pre_info['alpha_t_given_s'] / pre_info[
                    'sigma_t']) * eps_hat_pre
                sigma_pre = pre_info['sigma_t_given_s'] * pre_info['sigma_s'] / pre_info['sigma_t']
            elif zero_num == pre_info['xh'].size(0):
                zeros = torch.zeros(size=(pre_info['xh'].size(0), 1))
                gamma_0 = self.pre_edm.gamma(zeros).to(pre_info['xh'].device)
                sigma_pre = self.SNR(-0.5 * gamma_0).unsqueeze(1)
                mu_pre = self.compute_x_pred(eps_t=eps_hat_pre, z_t=pre_info['xh'], gamma_t=gamma_0)
            else:
                mu_pre = pre_info['xh'] / pre_info['alpha_t_given_s'] - (
                        pre_info['sigma2_t_given_s'] / pre_info['alpha_t_given_s'] / pre_info[
                    'sigma_t']) * eps_hat_pre
                sigma_pre = torch.nan_to_num(
                    pre_info['sigma_t_given_s'] * pre_info['sigma_s'] / pre_info['sigma_t'], nan=0.0,
                    posinf=1e10, neginf=-1e10)
                zeros = torch.zeros(size=(pre_info['xh'].size(0), 1))
                gamma_0 = self.pre_edm.gamma(zeros).to(pre_info['xh'].device)
                sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
                mu_x = self.compute_x_pred(eps_t=eps_hat_pre, z_t=pre_info['xh'], gamma_t=gamma_0)
                mu_pre = mu_pre * t_is_not_zero + mu_x * t_is_zero
                sigma_pre = sigma_pre * t_is_not_zero + sigma_x * t_is_zero
                # 
                # mu_pre = pre_info['xh'] / pre_info['alpha_t_given_s'] - (
                #         pre_info['sigma2_t_given_s'] / pre_info['alpha_t_given_s'] / pre_info[
                #     'sigma_t']) * eps_hat_pre
                # sigma_pre = torch.nan_to_num(pre_info['sigma_t_given_s'] * pre_info['sigma_s'] / pre_info['sigma_t'], nan=0.0, posinf=1e10, neginf=-1e10)
                # 

            z_s_pre = self.sample_normal(mu_pre, sigma_pre, pre_info['linker_mask'])
            z_s_pre = pre_info['xh'] * pre_info['fragment_mask'] + z_s_pre * pre_info['linker_mask']
            '''
            with torch.no_grad():
                t_is_zero = (t == 0.).float().unsqueeze(dim=-1)
                t_is_not_zero = 1. - t_is_zero
                s = t - 1./self.pre_edm.T
                z_s_pre = self.pre_edm.sample_p_zs_given_zt_only_linker(s, t, pre_info['xh'], pre_info['node_mask'],
                                                                        pre_info['fragment_mask'], pre_info['linker_mask'],
                                                                        pre_info['edge_mask'], pre_info['context'])

                # print(f"x coords: {pre_info['xh'][0][pre_info['mol_index'][0][0]][:3]}")
                # pre_info['z_s_pre'] = z_s_pre
                # torch.save(pre_info, f'datasets/case_sample/x_pre_input{int(t[0].item()*self.pre_edm.T)}.pt')

            # if sum(t_is_zero).item() > 0:
            #     x, h = self.pre_edm.sample_p_xh_given_z0_only_linker(pre_info['xh'], pre_info['node_mask'],
            #                                                              pre_info['fragment_mask'], pre_info['linker_mask'],
            #                                                              pre_info['edge_mask'], pre_info['context'])
            #     z0_s_pre =
            #     z_s_pre = t_is_not_zero*z_s_pre + t_is_zero*z0_s_pre

            # print(f"x_pre coords: {z_s_pre[0][pre_info['mol_index'][0][0]][:3]}")

            # =========================processing===============================
            for i, idx in enumerate(pre_info['mol_index']):
                if t_is_not_zero[i, 0, 0].item() == 1.:
                    xh_update[i, idx[1]: idx[2], :12] = self.w1*xh[i, idx[1]: idx[2], :12] + (1-self.w1)*z_s_pre[i, idx[0]: idx[0] + idx[2] - idx[1], :12]

        bs, n_nodes = xh.shape[0], xh.shape[1]
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if linker_mask is not None:
            linker_mask = linker_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        fragment_only_mask = context[..., -2].view(bs * n_nodes, 1)  # (B*N, 1)
        pocket_only_mask = context[..., -1].view(bs * n_nodes, 1)  # (B*N, 1)
        assert torch.all(fragment_only_mask.bool() | pocket_only_mask.bool() | linker_mask.bool() == node_mask.bool())

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)

        assert self.graph_type in ['4A', 'FC-4A', 'FC-10A-4A']
        if self.graph_type == '4A' or self.graph_type is None:
            edges = self.get_dist_edges_4A(x, node_mask, edge_mask)
        else:
            edges = self.get_dist_edges(x, node_mask, edge_mask, linker_mask, fragment_only_mask, pocket_only_mask)
        # edges = self.get_dist_edges(x, node_mask, edge_mask)
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=None
            )
            # print(f"x coords: {x_final[pre_info['mol_index'][0][1]]}")
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        if self.centering:
            vel = utils.remove_mean_with_mask(vel, node_mask)
        return torch.cat([vel, h_final], dim=2)

    def sample_normal(self, mu, sigma, node_mask):
        """Samples from a Normal distribution."""
        eps = self.sample_combined_position_feature_noise(mu.size(0), mu.size(1), node_mask)
        return mu + sigma * eps

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, mask):
        z_x = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=mask.device,
            node_mask=mask
        )
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, ATOM_TYPE),
            device=mask.device,
            node_mask=mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def compute_x_pred(self, eps_t, z_t, gamma_t):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.sigma(gamma_t, target_tensor=eps_t)
        alpha_t = self.alpha(gamma_t, target_tensor=eps_t)
        x_pred = 1. / alpha_t * (z_t - sigma_t * eps_t)
        return x_pred

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    @staticmethod
    def get_dist_edges_4A(x, node_mask, batch_mask):
        node_mask = node_mask.squeeze().bool()
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        dists_adj = (torch.cdist(x, x) <= 4)
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        adj = batch_adj & nodes_adj & dists_adj & rm_self_loops
        edges = torch.stack(torch.where(adj))
        return edges

    def get_dist_edges(self, x, node_mask, batch_mask, linker_mask, fragment_only_mask, pocket_only_mask):
        node_mask = node_mask.squeeze().bool()
        linker_mask = linker_mask.squeeze().bool() & node_mask
        fragment_only_mask = fragment_only_mask.squeeze().bool() & node_mask
        pocket_only_mask = pocket_only_mask.squeeze().bool() & node_mask
        ligand_mask = linker_mask | fragment_only_mask

        # General constrains:
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        constraints = batch_adj & nodes_adj & rm_self_loops

        # Ligand atoms – fully-connected graph
        ligand_adj = (ligand_mask[:, None] & ligand_mask[None, :])
        ligand_interactions = ligand_adj & constraints

        # Pocket atoms - within 4A
        pocket_adj = (pocket_only_mask[:, None] & pocket_only_mask[None, :])
        pocket_dists_adj = (torch.cdist(x, x) <= 4)
        pocket_interactions = pocket_adj & pocket_dists_adj & constraints

        # Pocket-ligand atoms - within 10A
        pocket_ligand_cutoff = 4 if self.graph_type == 'FC-4A' else 10
        pocket_ligand_adj = (ligand_mask[:, None] & pocket_only_mask[None, :])
        pocket_ligand_adj = pocket_ligand_adj | (pocket_only_mask[:, None] & ligand_mask[None, :])
        pocket_ligand_dists_adj = (torch.cdist(x, x) <= pocket_ligand_cutoff)
        pocket_ligand_interactions = pocket_ligand_adj & pocket_ligand_dists_adj & constraints

        adj = ligand_interactions | pocket_interactions | pocket_ligand_interactions
        edges = torch.stack(torch.where(adj))
        return edges

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

