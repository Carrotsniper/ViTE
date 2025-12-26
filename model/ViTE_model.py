import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv

from utils.RT_layer import RT, RTNoEdgeInit

class ExpertRouter(nn.Module):
    """
    Graph order MoE based on Top-P selection    
    Dynamically select first-order vs high-order graph interactions
    """
    def __init__(self, 
                 input_dim,           # input feature dimension
                 num_experts=2,       # number of experts: 2 (first-order + high-order)
                 top_p=0.7,          # Top-P threshold
                 noise_epsilon=1e-2,  # noise parameter
                 load_balance_weight=0.1):  # load balance weight
        super().__init__()
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.top_p = float(top_p)
        self.noise_epsilon = float(noise_epsilon)  # ensure conversion to float
        self.load_balance_weight = float(load_balance_weight)
        
        # gating network
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.noise = nn.Linear(input_dim, num_experts, bias=False)
        
        # activation function
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        
        # expert type identification (0: first-order, 1: high-order)
        self.expert_types = ['first_order', 'high_order']
        
    def cv_squared(self, x):
        """Squared coefficient of variation - for load balancing"""
        eps = 1e-10
        if x.shape[0] <= 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)
    
    def cross_entropy_loss(self, x):
        """Cross entropy loss - encourage expert selection diversity"""
        eps = 1e-10
        if x.shape[0] <= 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=-1).mean()
    
    def noisy_top_p_gating(self, x, is_training=True):
        """
        Noisy Top-P gating mechanism
        
        Args:
            x: [B, N, D] or [B*N, D] - node features
            is_training: whether in training mode
            
        Returns:
            expert_mask: [B, N, num_experts] - expert selection mask
            gating_loss: gating loss
        """
        original_shape = x.shape
        if len(x.shape) == 3:
            B, N, D = x.shape
            x_flat = x.view(B * N, D)
        else:
            x_flat = x
            B, N = 1, x.shape[0]
        
        # 1. Compute basic gating scores
        clean_logits = self.gate(x_flat)  # [B*N, num_experts]
        
        # 2. Add noise (only during training)
        if is_training:
            raw_noise = self.noise(x_flat)
            noise_stddev = self.softplus(raw_noise) + self.noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
                
        # Convert logits to probabilities
        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy_loss(logits)

        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        
        # Handle indices based on tensor dimensions
        if len(mask.shape) == 2:  # 2D case: (B*N) x num_experts
            top_p_mask[zero_indices[0], sorted_indices[zero_indices[0], zero_indices[1]]] = 1
        else:  # 3D case: B x N x num_experts  
            top_p_mask[zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1

        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1
        loss = loss_importance + lambda_2 * loss_dynamic

        return top_p_mask, loss
    
    def forward(self, x, return_gating_info=False):
        """
        Forward propagation
        
        Args:
            x: [B, N, D] - node features
            return_gating_info: whether to return gating info
            
        Returns:
            expert_weights: [B, N, num_experts] - expert weights
            expert_decisions: [B, N] - expert usage pattern (0=use 1 expert, 1=use 2 experts)
            gating_loss: gating loss
            gating_info: (optional) detailed gating information
        """
        # Record original shape
        B, N, D = x.shape
        
        # 1. Gating decision
        expert_weights, gating_loss = self.noisy_top_p_gating(x, self.training)
        
        # 2. Ensure expert_weights shape is correct [B, N, num_experts]
        if expert_weights.dim() == 2:
            expert_weights = expert_weights.view(B, N, self.num_experts)
        
        # 3. Determine expert usage pattern: 0=use 1 expert, 1=use 2 experts
        # Count how many expert weights > 0 for each node
        active_experts_count = (expert_weights > 0).sum(dim=-1)  # [B, N]
        expert_decisions = (active_experts_count > 1).long()  # [B, N]
        
        # 3. Statistics
        single_expert_ratio = (expert_decisions == 0).float().mean()
        mixed_expert_ratio = (expert_decisions == 1).float().mean()
        
        if return_gating_info:
            gating_info = {
                'expert_weights': expert_weights,
                'expert_decisions': expert_decisions,
                'single_expert_ratio': single_expert_ratio,
                'mixed_expert_ratio': mixed_expert_ratio,
                'gating_loss': gating_loss,
                'selected_experts_per_node': (expert_weights > 0).sum(dim=-1).float().mean()
            }
            return expert_weights, expert_decisions, gating_loss, gating_info
        
        return expert_weights, expert_decisions, gating_loss


class ExpertProcessor(nn.Module):
    """
    Works with ExpertRouter to execute actual graph operations
    """
    def __init__(self, input_dim, hidden_dim=128, num_virtual_nodes=5, edge_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # First-order graph expert: two-layer GNN with edge updates
        self.first_order_expert = OneHopExpert(input_dim, hidden_dim, edge_dim)
        
        # High-order graph expert: use virtual nodes to capture high-order interactions
        self.high_order_expert = HighOrderExpert(input_dim, hidden_dim, num_virtual_nodes)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, input_dim)
        
        # Add residual normalization
        self.residual_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, edge_index, edge_attr, expert_weights, expert_decisions, batch_size):
        """
        Args:
            x: [B*N, D] node features
            edge_index: [2, E] edge index
            edge_attr: [E,] edge weights
            expert_weights: [B, N, 2] expert weights
            expert_decisions: [B, N] dominant expert decision
            batch_size: batch size
            
        Returns:
            output: [B*N, D] processed features
        """
        num_nodes_per_batch = x.size(0) // batch_size
        
        # Flatten weights for computation
        expert_weights_flat = expert_weights.view(-1, 2)  # [B*N, 2]
        
        # Compute outputs from both experts separately
        first_order_out = self.first_order_expert(x, edge_index, edge_attr)
        high_order_out = self.high_order_expert(x, edge_index, edge_attr, 
                                                batch_size=batch_size, 
                                                num_nodes_per_batch=num_nodes_per_batch)
        
        # Combine outputs according to expert weights
        first_order_weight = expert_weights_flat[:, 0:1]  # [B*N, 1]
        high_order_weight = expert_weights_flat[:, 1:2]   # [B*N, 1]
        
        # Weighted combination
        weighted_output = first_order_weight * first_order_out + high_order_weight * high_order_out
        
        # Output projection + residual connection
        output = self.output_proj(weighted_output)
        output = self.residual_norm(output + x)  # Add residual connection and normalization
        
        return output


class OneHopExpert(nn.Module):
    """First-order graph expert: two-layer GNN with edge updates"""
    def __init__(self, input_dim, hidden_dim=128, edge_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Two-layer GCN
        self.gcn1 = GCNConv(input_dim, input_dim)
        self.gcn2 = GCNConv(input_dim, input_dim)
        
        # Node layer normalization
        self.norm1_n = nn.LayerNorm(input_dim)
        self.norm2_n = nn.LayerNorm(input_dim)
        
        # Edge feature initialization (from node feature similarity)
        self.edge_init_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, edge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(edge_dim, edge_dim)
        )
        
        # Edge update network (simplified version referencing prt.py)
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        # Edge normalization
        self.norm_e = nn.LayerNorm(edge_dim)
        
        # Mapping from edge features to weights
        self.edge_to_weight = nn.Sequential(
            nn.Linear(edge_dim, 1),
            nn.Sigmoid()
        )
        
        # Activation function
        self.activation = nn.GELU()
        
    def compute_edge_features(self, x, edge_index):
        """Compute initial edge features (based on node feature similarity)"""
        row, col = edge_index
        # Get source and target node features connected by edges
        source_nodes = x[row]  # [E, D]
        target_nodes = x[col]  # [E, D]
        
        # Concatenate source and target node features
        edge_input = torch.cat([source_nodes, target_nodes], dim=-1)  # [E, 2*D]
        
        # Generate edge features through MLP
        edge_features = self.edge_init_mlp(edge_input)  # [E, edge_dim]
        
        return edge_features
        
    def update_edge_features(self, x, edge_index, edge_features):
        """Update edge features (simplified version referencing prt.py)"""
        row, col = edge_index
        source_nodes = x[row]  # [E, D] source node features
        target_nodes = x[col]  # [E, D] target node features
        
        # Concatenate source nodes, target nodes and current edge features
        concatenated_inputs = torch.cat([source_nodes, target_nodes, edge_features], dim=-1)  # [E, 2*D + edge_dim]
        
        # Update edge features through MLP
        updated_edge_features = self.edge_update_mlp(concatenated_inputs)  # [E, edge_dim]
        
        # Residual connection + normalization
        edge_features = edge_features + updated_edge_features
        edge_features = self.norm_e(edge_features)
        
        return edge_features
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, D] node features
            edge_index: [2, E] edge index
            edge_attr: [E,] edge weights (optional, use similarity calculation if not provided)
        """
        # 1. Initialize edge features (based on node feature similarity)
        edge_features = self.compute_edge_features(x, edge_index)  # [E, edge_dim]
        
        # 2. First layer GCN
        # Convert edge features to weights
        edge_weights = self.edge_to_weight(edge_features).squeeze(-1)  # [E,]
        out1 = self.gcn1(x, edge_index, edge_weights)
        out1 = self.activation(out1)
        out1 = self.norm1_n(out1)
        
        # 3. Edge feature update (based on first layer node features)
        edge_features = self.update_edge_features(out1, edge_index, edge_features)
        
        # 4. Second layer GCN
        # Recompute edge weights
        edge_weights = self.edge_to_weight(edge_features).squeeze(-1)  # [E,]
        out2 = self.gcn2(out1, edge_index, edge_weights)
        out2 = self.activation(out2)
        out2 = self.norm2_n(out2)
        
        return out2


class HighOrderExpert(nn.Module):
    """High-order graph expert: VirtualGraph for qe1 high-order interactions"""
    def __init__(self, input_dim, hidden_dim=128, num_virtual_nodes=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_virtual_nodes = num_virtual_nodes
        
        # Structured virtual node initialization
        self.virtual_nodes = self._init_diverse_virtual_nodes(num_virtual_nodes, input_dim)
        
        # Virtual node enhanced graph neural network
        self.gat = GATConv(input_dim, input_dim, heads=4, dropout=0.1, concat=False)
        self.gcn = GCNConv(input_dim, input_dim)
        
        # Normalization layer
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Activation function
        self.activation = nn.GELU()
        
    def _init_diverse_virtual_nodes(self, num_virtual_nodes, model_dim):
        """Create virtual nodes with different characteristics"""
        virtual_nodes = []
        for i in range(num_virtual_nodes):
            if i == 0:
                node = torch.randn(model_dim) * 0.5
                node[:model_dim//4] = 1.0  
            elif i == 1:
                node = torch.randn(model_dim) * 0.5
                node[model_dim//4:model_dim//2] = -1.0  
            elif i == 2:
                node = torch.randn(model_dim) * 0.5
                node[model_dim//2:3*model_dim//4] = 0.5  
            else:
                node = torch.randn(model_dim) * 0.3   
            virtual_nodes.append(node)
        return nn.Parameter(torch.stack(virtual_nodes, dim=0))
    
    def build_two_stage_graphs(self, x, edge_index, edge_attr, batch_size):
        """Build two-stage graph structure"""
        device = x.device
        num_nodes_per_batch = x.size(0) // batch_size
        total_real_nodes = x.size(0) 
        virtual_nodes_all_batches = []
        for b in range(batch_size):
            batch_virtual_nodes = self.virtual_nodes.clone()  # [V, D]
            virtual_nodes_all_batches.append(batch_virtual_nodes)
        
        all_virtual_nodes = torch.cat(virtual_nodes_all_batches, dim=0)  # [B*V, D]
        all_nodes = torch.cat([x, all_virtual_nodes], dim=0)  # [B*(N+V), D]
        
        stage1_edge_list = []
        stage1_edge_attr_list = []
        stage2_edge_list = []
        stage2_edge_attr_list = []
        
        for b in range(batch_size):
            # Node index range for current batch
            real_start = b * num_nodes_per_batch
            real_end = (b + 1) * num_nodes_per_batch
            virtual_start = total_real_nodes + b * self.num_virtual_nodes
            virtual_end = total_real_nodes + (b + 1) * self.num_virtual_nodes
            
            # First stage: real nodes -> virtual nodes
            for i in range(real_start, real_end):  # real nodes
                for j in range(virtual_start, virtual_end):  # virtual nodes
                    stage1_edge_list.append(torch.tensor([[i], [j]], device=device))
                    stage1_edge_attr_list.append(torch.ones(1, device=device))
            # Second stage: virtual nodes -> real nodes  
            for i in range(virtual_start, virtual_end):  # virtual nodes
                for j in range(real_start, real_end):  # real nodes
                    stage2_edge_list.append(torch.tensor([[i], [j]], device=device))
                    stage2_edge_attr_list.append(torch.ones(1, device=device))
        # Merge first stage edges
        if stage1_edge_list:
            stage1_edge_index = torch.cat(stage1_edge_list, dim=1)
            stage1_edge_attr = torch.cat(stage1_edge_attr_list, dim=0)
        else:
            stage1_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            stage1_edge_attr = torch.empty((0,), dtype=torch.float, device=device)
        
        # Merge second stage edges
        if stage2_edge_list:
            stage2_edge_index = torch.cat(stage2_edge_list, dim=1)
            stage2_edge_attr = torch.cat(stage2_edge_attr_list, dim=0)
        else:
            stage2_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            stage2_edge_attr = torch.empty((0,), dtype=torch.float, device=device)
        
        return all_nodes, stage1_edge_index, stage1_edge_attr, stage2_edge_index, stage2_edge_attr
    
    def forward(self, x, edge_index, edge_attr=None, batch_size=None, num_nodes_per_batch=None):
        """
        Args:
            x: [B*N, D] flattened real node features
            edge_index: [2, E] original edge index
            edge_attr: [E,] original edge weights
            batch_size: batch size
            num_nodes_per_batch: number of nodes per batch
            
        Returns:
            output: [B*N, D] enhanced real node features (remove virtual nodes)
        """
        total_real_nodes = x.size(0)  # B*N
        
        # If batch info not provided, try to infer (assume single batch)
        if batch_size is None or num_nodes_per_batch is None:
            batch_size = 1
            num_nodes_per_batch = total_real_nodes
        
        # 1. Build two-stage graph structure
        all_nodes, stage1_edge_index, stage1_edge_attr, stage2_edge_index, stage2_edge_attr = self.build_two_stage_graphs(
            x, edge_index, edge_attr, batch_size
        )
        
        # 2. First layer GAT: real nodes → virtual nodes (information aggregation stage)
        gat_out = self.gat(all_nodes, stage1_edge_index)
        gat_out = self.activation(gat_out)
        gat_out = self.norm1(gat_out)
        
        # 3. Second layer GCN: virtual nodes → real nodes (information distribution stage)  
        gcn_out = self.gcn(gat_out, stage2_edge_index)
        gcn_out = self.activation(gcn_out)
        gcn_out = self.norm2(gcn_out)
        
        # 4. Only return real node features (remove virtual nodes)
        real_node_features = gcn_out[:total_real_nodes]
        
        return real_node_features


class GraphStructureBuilder(nn.Module):
    """Graph structure builder: dynamically build different types of graphs"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
    def build_similarity_graph(self, x, threshold=0.15, k_nearest=4):
        """Build graph based on feature similarity"""
        # Compute similarity matrix
        x_norm = F.normalize(x, p=2, dim=-1)
        sim_matrix = torch.mm(x_norm, x_norm.t())
        
        # Method 1: threshold filtering
        adj_threshold = (sim_matrix > threshold).float()
        adj_threshold = adj_threshold - torch.eye(x.size(0), device=x.device)
        
        # Method 2: k-nearest neighbors
        # Ensure k value doesn't exceed available node count
        num_nodes = x.size(0)
        actual_k = min(k_nearest + 1, num_nodes)  # +1 because including self
        
        if actual_k > 1:  # Need at least 2 nodes to build edges
            _, top_k_indices = torch.topk(sim_matrix, actual_k, dim=-1)
            adj_knn = torch.zeros_like(sim_matrix)
            for i in range(num_nodes):
                # Exclude self (index 0 is usually self, as self-similarity is highest)
                neighbors = top_k_indices[i, 1:actual_k] if actual_k > 1 else []
                if len(neighbors) > 0:
                    adj_knn[i, neighbors] = 1
        else:
            # If only 1 node, create empty adjacency matrix
            adj_knn = torch.zeros_like(sim_matrix)
        
        # Combine two methods
        adj_matrix = adj_threshold * adj_knn
        edge_index = adj_matrix.nonzero().t()
        
        if edge_index.numel() > 0:
            edge_attr = sim_matrix[edge_index[0], edge_index[1]]
        else:
            edge_attr = torch.empty((0,), dtype=torch.float, device=x.device)
        
        return edge_index, edge_attr
    
    def build_batch_graph(self, x, batch_info):
        """Build graph structure for batch data"""
        edge_indices = []
        edge_attrs = []
        
        unique_batches = torch.unique(batch_info)
        node_offset = 0
        
        for batch_id in unique_batches:
            # Get nodes of current batch
            batch_mask = (batch_info == batch_id)
            batch_nodes = x[batch_mask]
            
            # Build graph for current batch
            batch_edge_index, batch_edge_attr = self.build_similarity_graph(batch_nodes)
            
            # Adjust node indices
            batch_edge_index = batch_edge_index + node_offset
            
            edge_indices.append(batch_edge_index)
            edge_attrs.append(batch_edge_attr)
            
            node_offset += batch_nodes.size(0)
        
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_attr = torch.cat(edge_attrs, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_attr = torch.empty((0,), dtype=torch.float, device=x.device)
        
        return edge_index, edge_attr


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(1024, 512), activation='relu'):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_dims)
        dims.append(output_dim)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe[None].repeat(num_a, 1, 1)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset) #(N, T, D)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x) #(N, T, D)


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.multiplier = 3
        self.decoder_mlp = MLP(
            args.model_dim*self.multiplier,
            args.future_length*2,
            hidden_dims=(
                args.decoder_hidden_dim,
                args.decoder_hidden_dim // 2
            )
        )
    def forward(self, final_feature, cur_location):
        outputs = self.decoder_mlp(final_feature)         
        outputs = outputs.view(-1, self.args.future_length, 2)
        if not self.args.pred_rel:
            outputs = outputs + cur_location
        
        return outputs        


class ViTE(nn.Module):
    def __init__(self, args):
        super(ViTE, self).__init__()
        self.args = args
        module_args = {
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'node_dim': args.model_dim,
            'node_hidden_dim': args.hidden_dim,
            'edge_dim': args.model_dim,
            'edge_hidden_dim_1': args.hidden_dim,
            'edge_hidden_dim_2': args.hidden_dim,
            'dropout': args.dropout,
            'topk': 6,
            'threshold': 0.12,
        }
        
        self.input_dim = len(args.inputs)
        self.input_fc = nn.Linear(self.input_dim, args.model_dim)
        self.input_fc2 = nn.Linear(args.model_dim*args.past_length, args.model_dim)
        self.pos_encoder = PositionalAgentEncoding(args.model_dim, 0.1, concat=True)
        self.pair_encoders = nn.ModuleList()
        for i in range(args.num_layers):
            if i == 0:
                self.pair_encoders.append(RT(**module_args))
            else:
                self.pair_encoders.append(RTNoEdgeInit(**module_args))
        
        self.graph_builder = GraphStructureBuilder(args.model_dim)
        self.moe_layers = nn.ModuleList()
        self.graph_processors = nn.ModuleList()
        num_moe_layers = getattr(args, 'num_moe_layers', 2)
        for _ in range(num_moe_layers):
            moe_gating = ExpertRouter(
                input_dim=args.model_dim,
                num_experts=getattr(args, 'num_experts', 2),
                top_p=getattr(args, 'top_p', 0.5),
                noise_epsilon=getattr(args, 'noise_epsilon', 1e-2),
                load_balance_weight=getattr(args, 'load_balance_weight', 0.2)
            )
            
            graph_processor = ExpertProcessor(
                input_dim=args.model_dim,
                hidden_dim=getattr(args, 'moe_hidden_dim', 128),
                num_virtual_nodes=getattr(args, 'num_virtual_nodes', 5),
                edge_dim=getattr(args, 'edge_dim', 64)
            )
            
            self.moe_layers.append(moe_gating)
            self.graph_processors.append(graph_processor)
        
        # Create multiple decoder heads
        for i in range(args.sample_k):
            self.add_module("head_%d" % i, Decoder(args))
        
    def forward(self, x_abs, x_rel, log_moe_info=False):
        inputs = []
        batch_size, num_agents, length, _ = x_abs.shape
        cur_pos = x_abs[:, :, [-1]].view(batch_size*num_agents, 1, -1).contiguous()
                
        if 'pos_x' in self.args.inputs and 'pos_y' in self.args.inputs:
            inputs.append(x_abs)
        if 'vel_x' in self.args.inputs and 'vel_y' in self.args.inputs:
            inputs.append(x_rel)
        
        inputs = torch.cat(inputs, dim=-1)
        inputs = inputs.view(batch_size*num_agents, length, -1).contiguous()                                         # N x T_obs x 2
        inputs_fc = self.input_fc(inputs).view(batch_size*num_agents, length, self.args.model_dim)                   # N x T_obs x D=64
        inputs_pos = self.pos_encoder(inputs_fc, num_a=batch_size*num_agents)                                        # N x T_obs x D=64
        inputs_pos = inputs_pos.view(batch_size, num_agents, length, self.args.model_dim)                            # B x N x T_obs x D=64
        n_initial = self.input_fc2(inputs_pos.contiguous().view(batch_size, num_agents, length*self.args.model_dim)) # B x N x (T_obs*D=64) = 512->64
        
        pair_emb, e_pair = n_initial, None                # n_initial: B x N x 64
        
        for i in range(self.args.num_layers):
            pair_emb, e_pair = self.pair_encoders[i](pair_emb, e_pair, return_edge=True)                           
        
        n_moe = n_initial
        total_moe_loss = 0
        moe_features = []
        all_gating_info = []  # New: collect gating information from all layers
        
        # Flatten for graph processing
        n_moe_flat = n_moe.view(batch_size * num_agents, self.args.model_dim)
        
        # Create batch information
        batch_info = torch.arange(batch_size, device=n_moe_flat.device).repeat_interleave(num_agents)
        
        # Build graph structure
        edge_index, edge_attr = self.graph_builder.build_batch_graph(n_moe_flat, batch_info)
        
        current_features = n_moe_flat
        current_features_batch = current_features.view(batch_size, num_agents, self.args.model_dim)
        
        for i, (moe_gating, graph_processor) in enumerate(zip(self.moe_layers, self.graph_processors)):
            # 1. MoE gating decision - modify here
            if log_moe_info:
                expert_weights, expert_decisions, moe_loss, gating_info = moe_gating(
                    current_features_batch, return_gating_info=True
                )
                # Add layer information
                gating_info['layer_id'] = i
                all_gating_info.append(gating_info)
            else:
                expert_weights, expert_decisions, moe_loss = moe_gating(current_features_batch)
            
            # 2. Graph processing
            enhanced_features = graph_processor(
                current_features, edge_index, edge_attr, expert_weights, expert_decisions, batch_size
            )
            
            moe_features.append(enhanced_features.view(batch_size, num_agents, self.args.model_dim))
            total_moe_loss += moe_loss
            
            current_features = enhanced_features
            current_features_batch = current_features.view(batch_size, num_agents, self.args.model_dim)
        router_emb = moe_features[-1]

        
        n_final = torch.cat([n_initial, pair_emb, router_emb], dim=-1) 
        
        out_list = []
        for i in range(self.args.sample_k):
            out = self._modules["head_%d" % i](n_final, cur_pos)
            out_list.append(out[:, None, :, :])
        
        out = torch.cat(out_list, dim=2)                         
        out = out.view(batch_size, num_agents, self.args.sample_k, self.args.future_length, -1)
        
        # Modify return value
        if log_moe_info:
            return out, total_moe_loss, all_gating_info
        else:
            return out, total_moe_loss
    
