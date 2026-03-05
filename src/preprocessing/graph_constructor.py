"""
Advanced Graph Construction for Cyber Network Analysis
Implements multiple graph construction strategies with temporal modeling,
feature caching, interactive visualization, and flow/hierarchical graphs.

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks
"""

import hashlib
import json
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedGraphConstructor:
    """
    Advanced graph construction with multiple strategies for cyber network modeling.

    Supported graph types:
    - host_based:    Nodes = IP addresses, edges = traffic between hosts
    - flow_based:    Nodes = individual flows, edges = shared endpoint
    - hierarchical:  Multi-level (subnet -> host -> port)
    - temporal:      Time-evolving graph snapshots

    New in this version:
    - Graph feature caching to avoid recomputing expensive centrality measures
    - Interactive HTML export via pyvis (Phase 4)
    - Flow-based and hierarchical construction (Phase 5)
    """

    def __init__(self,
                 construction_method: str = 'host_based',
                 time_window: int = 300,
                 directed: bool = True,
                 self_loops: bool = False,
                 cache_dir: str = 'data/cache'):
        """
        Initialize graph constructor.

        Args:
            construction_method: 'host_based', 'flow_based', or 'hierarchical'
            time_window: Time window in seconds for temporal snapshots
            directed: Create directed graph
            self_loops: Add self-loops to nodes
            cache_dir: Directory to store cached graph features
        """
        self.construction_method = construction_method
        self.time_window = time_window
        self.directed = directed
        self.self_loops = self_loops
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Mappings
        self.node_id_map: Dict[str, int] = {}
        self.reverse_node_map: Dict[int, str] = {}

        # Statistics
        self.graph_stats = {
            'construction_method': construction_method,
            'total_nodes': 0,
            'total_edges': 0,
            'avg_degree': 0,
            'density': 0,
            'components': 0
        }

        logger.info(f"Initialized AdvancedGraphConstructor with method: {construction_method}")

    # ------------------------------------------------------------------
    # Caching helpers (Phase 2)
    # ------------------------------------------------------------------

    def _cache_key(self, df: pd.DataFrame) -> str:
        """Generate a cache key based on DataFrame shape and content hash."""
        shape_str = f"{len(df)}_{list(df.columns)}"
        sample = df.head(100).to_json()
        return hashlib.md5((shape_str + sample).encode()).hexdigest()

    def _cache_path(self, key: str, suffix: str = 'node_features') -> Path:
        return self.cache_dir / f"graph_{suffix}_{key}.pkl"

    def _load_from_cache(self, key: str, suffix: str = 'node_features') -> Optional[object]:
        path = self._cache_path(key, suffix)
        if path.exists():
            logger.info(f"Loading cached {suffix} from {path}")
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, obj: object, key: str, suffix: str = 'node_features') -> None:
        path = self._cache_path(key, suffix)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Cached {suffix} to {path}")

    # ------------------------------------------------------------------
    # Node mapping
    # ------------------------------------------------------------------

    def _create_node_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        """Create mapping from IP addresses to node IDs."""
        if self.construction_method == 'host_based':
            unique_ips = pd.concat([df['src_ip'], df['dst_ip']]).dropna().unique()
            self.node_id_map = {ip: idx for idx, ip in enumerate(unique_ips)}
            self.reverse_node_map = {idx: ip for ip, idx in self.node_id_map.items()}

        logger.info(f"Created node mapping with {len(self.node_id_map)} unique nodes")
        return self.node_id_map

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------

    def _compute_node_features(self, df: pd.DataFrame, G: nx.Graph = None,
                                use_cache: bool = True) -> torch.Tensor:
        """
        Compute comprehensive node features with optional disk caching.

        Features per node (15):
        - Out/in/total degree
        - Attack involvement (src, dst, ratio)
        - Traffic volume (fwd packets, bwd packets, avg flow duration)
        - Port diversity (unique dst ports, unique src ports)
        - Graph-theoretic (betweenness, closeness, clustering)
        - Degree ratio
        """
        cache_key = self._cache_key(df)
        if use_cache:
            cached = self._load_from_cache(cache_key, 'node_features')
            if cached is not None:
                return cached

        logger.info("Computing node features...")

        node_features_list = []

        for node_id in range(len(self.node_id_map)):
            if node_id not in self.reverse_node_map:
                node_features_list.append([0.0] * 15)
                continue

            ip = self.reverse_node_map[node_id]

            src_flows = df[df['src_ip'] == ip]
            dst_flows = df[df['dst_ip'] == ip]

            out_degree = len(src_flows)
            in_degree = len(dst_flows)
            total_degree = out_degree + in_degree

            attack_as_src = src_flows['is_attack'].sum() if 'is_attack' in df.columns else 0
            attack_as_dst = dst_flows['is_attack'].sum() if 'is_attack' in df.columns else 0
            attack_ratio = (attack_as_src + attack_as_dst) / total_degree if total_degree > 0 else 0

            total_fwd_packets = src_flows['Total Fwd Packets'].sum() if 'Total Fwd Packets' in df.columns else 0
            total_bwd_packets = dst_flows['Total Backward Packets'].sum() if 'Total Backward Packets' in df.columns else 0
            avg_flow_duration = src_flows['Flow Duration'].mean() if 'Flow Duration' in df.columns else 0

            unique_dst_ports = src_flows['dst_port'].nunique() if 'dst_port' in df.columns else 0
            unique_src_ports = dst_flows['src_port'].nunique() if 'src_port' in df.columns else 0

            if G is not None and node_id in G.nodes():
                try:
                    betweenness = nx.betweenness_centrality(G, normalized=True).get(node_id, 0)
                    closeness = nx.closeness_centrality(G).get(node_id, 0)
                    clustering = nx.clustering(G).get(node_id, 0)
                except Exception:
                    betweenness = closeness = clustering = 0.0
            else:
                betweenness = closeness = clustering = 0.0

            features = [
                float(out_degree),
                float(in_degree),
                float(total_degree),
                float(attack_ratio),
                float(attack_as_src),
                float(attack_as_dst),
                float(total_fwd_packets),
                float(total_bwd_packets),
                float(avg_flow_duration if not np.isnan(avg_flow_duration) else 0.0),
                float(unique_dst_ports),
                float(unique_src_ports),
                float(betweenness),
                float(closeness),
                float(clustering),
                float(out_degree / (in_degree + 1))
            ]
            node_features_list.append(features)

        node_features = torch.tensor(node_features_list, dtype=torch.float)
        logger.info(f"Node features computed. Shape: {node_features.shape}")

        if use_cache:
            self._save_to_cache(node_features, cache_key, 'node_features')

        return node_features

    def _compute_edge_features(self, edge_df: pd.DataFrame) -> torch.Tensor:
        """Compute edge features from aggregated flows."""
        edge_features = []

        for _, row in edge_df.iterrows():
            features = [
                float(row.get('total_fwd_packets', 0)),
                float(row.get('total_bwd_packets', 0)),
                float(row.get('total_bytes', 0)),
                float(row.get('avg_flow_duration', 0)),
                float(row.get('num_flows', 1)),
                float(row.get('is_attack', 0)),
                float(row.get('max_packet_length', 0)),
                float(row.get('flow_bytes_per_sec', 0))
            ]
            edge_features.append(features)

        return torch.tensor(edge_features, dtype=torch.float)

    # ------------------------------------------------------------------
    # Host-based graph (original method)
    # ------------------------------------------------------------------

    def build_networkx_graph(self,
                             df: pd.DataFrame,
                             aggregate_edges: bool = True,
                             min_flows: int = 1) -> nx.Graph:
        """
        Build NetworkX host-based graph.

        Args:
            df: DataFrame with network flow data (must have 'src_ip', 'dst_ip')
            aggregate_edges: Aggregate multiple flows between same hosts
            min_flows: Minimum number of flows to create an edge

        Returns:
            NetworkX Graph or DiGraph
        """
        logger.info(f"Building NetworkX graph using {self.construction_method} method...")

        if self.construction_method == 'flow_based':
            return self.build_flow_based_graph(df)
        if self.construction_method == 'hierarchical':
            return self.build_hierarchical_graph(df)

        G = nx.DiGraph() if self.directed else nx.Graph()

        self._create_node_mapping(df)

        for ip, node_id in self.node_id_map.items():
            G.add_node(node_id, ip=ip)

        if aggregate_edges:
            logger.info("Aggregating flows between host pairs...")

            agg_dict = {
                'Total Fwd Packets': 'sum',
                'Total Backward Packets': 'sum',
                'Flow Duration': 'mean',
                'Flow Bytes/s': 'mean',
                'is_attack': 'max',
            }
            agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

            edge_data = df.groupby(['src_ip', 'dst_ip']).agg(agg_dict).reset_index()
            edge_data['num_flows'] = df.groupby(['src_ip', 'dst_ip']).size().values

            edge_data = edge_data[edge_data['num_flows'] >= min_flows]
            logger.info(f"Aggregated {len(df)} flows into {len(edge_data)} edges")
        else:
            edge_data = df

        for _, row in edge_data.iterrows():
            src_id = self.node_id_map.get(row['src_ip'])
            dst_id = self.node_id_map.get(row['dst_ip'])

            if src_id is None or dst_id is None:
                continue

            edge_attrs = {
                'weight': float(row.get('num_flows', 1)),
                'is_attack': int(row.get('is_attack', 0)),
                'total_packets': float(
                    row.get('Total Fwd Packets', 0) + row.get('Total Backward Packets', 0)
                ),
                'flow_duration': float(row.get('Flow Duration', 0)),
                'bytes_per_sec': float(row.get('Flow Bytes/s', 0))
            }
            G.add_edge(src_id, dst_id, **edge_attrs)

        self._update_graph_stats(G)
        return G

    # ------------------------------------------------------------------
    # Flow-based graph (Phase 5)
    # ------------------------------------------------------------------

    def build_flow_based_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build a flow-based graph where each node is an individual network flow.

        Nodes represent flows; edges connect flows sharing a source or
        destination IP, enabling fine-grained attack signature detection.

        Args:
            df: DataFrame with flow records

        Returns:
            NetworkX DiGraph (or Graph)
        """
        logger.info("Building flow-based graph...")

        G = nx.DiGraph() if self.directed else nx.Graph()

        # Each row is a node
        for idx, row in df.iterrows():
            node_attrs = {
                'src_ip': row.get('src_ip', ''),
                'dst_ip': row.get('dst_ip', ''),
                'is_attack': int(row.get('is_attack', 0)),
                'label_id': int(row.get('label_id', 0)),
            }
            G.add_node(idx, **node_attrs)

        # Build index maps for fast lookup
        src_ip_index: Dict[str, List[int]] = defaultdict(list)
        dst_ip_index: Dict[str, List[int]] = defaultdict(list)

        for idx, row in df.iterrows():
            if 'src_ip' in df.columns:
                src_ip_index[row['src_ip']].append(idx)
            if 'dst_ip' in df.columns:
                dst_ip_index[row['dst_ip']].append(idx)

        # Connect flows sharing the same destination IP (potential attack chain)
        max_edges_per_node = 20  # cap to avoid dense cliques on scanning IPs
        for ip, indices in dst_ip_index.items():
            for i in range(len(indices)):
                for j in range(i + 1, min(i + max_edges_per_node, len(indices))):
                    G.add_edge(indices[i], indices[j], edge_type='shared_dst')

        # Connect flows where one flow's destination is another's source
        for ip, src_indices in src_ip_index.items():
            for src_idx in src_indices:
                for dst_idx in dst_ip_index.get(ip, []):
                    if src_idx != dst_idx:
                        G.add_edge(dst_idx, src_idx, edge_type='forward_chain')

        # Update node mappings (node_id = dataframe index)
        self.node_id_map = {str(idx): idx for idx in df.index}
        self.reverse_node_map = {idx: str(idx) for idx in df.index}

        self._update_graph_stats(G)
        logger.info(f"Flow-based graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # ------------------------------------------------------------------
    # Hierarchical graph (Phase 5)
    # ------------------------------------------------------------------

    def build_hierarchical_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build a hierarchical graph with three levels: subnet → host → port.

        Level 0 (subnet) nodes aggregate /24 subnets.
        Level 1 (host) nodes are individual IP addresses.
        Level 2 (port) nodes are (IP, port) pairs.

        Args:
            df: DataFrame with flow records

        Returns:
            NetworkX graph with hierarchical node types
        """
        logger.info("Building hierarchical graph (subnet -> host -> port)...")

        G = nx.DiGraph() if self.directed else nx.Graph()
        node_counter = 0

        subnet_map: Dict[str, int] = {}
        host_map: Dict[str, int] = {}
        port_map: Dict[str, int] = {}

        def get_subnet(ip: str) -> str:
            parts = str(ip).split('.')
            return '.'.join(parts[:3]) + '.0/24' if len(parts) == 4 else ip

        for _, row in df.iterrows():
            for ip_col, port_col in [('src_ip', 'src_port'), ('dst_ip', 'dst_port')]:
                ip = row.get(ip_col)
                port = row.get(port_col)
                if pd.isna(ip):
                    continue

                subnet = get_subnet(str(ip))

                # Subnet node
                if subnet not in subnet_map:
                    subnet_map[subnet] = node_counter
                    G.add_node(node_counter, level=0, label=subnet, node_type='subnet')
                    node_counter += 1

                # Host node
                if ip not in host_map:
                    host_map[ip] = node_counter
                    G.add_node(node_counter, level=1, label=str(ip), node_type='host')
                    node_counter += 1
                    G.add_edge(subnet_map[subnet], host_map[ip], edge_type='contains')

                # Port node
                if port is not None and not (isinstance(port, float) and np.isnan(port)):
                    port_key = f"{ip}:{int(port)}"
                    if port_key not in port_map:
                        port_map[port_key] = node_counter
                        G.add_node(node_counter, level=2, label=port_key, node_type='port')
                        node_counter += 1
                        G.add_edge(host_map[ip], port_map[port_key], edge_type='has_port')

        # Connect src_port to dst_port for each flow
        for _, row in df.iterrows():
            src_ip = row.get('src_ip')
            dst_ip = row.get('dst_ip')
            src_port = row.get('src_port')
            dst_port = row.get('dst_port')

            if pd.isna(src_ip) or pd.isna(dst_ip):
                continue

            src_key = f"{src_ip}:{int(src_port)}" if src_port and not np.isnan(float(src_port)) else None
            dst_key = f"{dst_ip}:{int(dst_port)}" if dst_port and not np.isnan(float(dst_port)) else None

            if src_key in port_map and dst_key in port_map:
                G.add_edge(
                    port_map[src_key],
                    port_map[dst_key],
                    edge_type='flow',
                    is_attack=int(row.get('is_attack', 0))
                )

        # Store combined map for downstream compatibility
        self.node_id_map = {**{k: v for k, v in subnet_map.items()},
                            **{k: v for k, v in host_map.items()},
                            **{k: v for k, v in port_map.items()}}
        self.reverse_node_map = {v: k for k, v in self.node_id_map.items()}

        self._update_graph_stats(G)
        logger.info(f"Hierarchical graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # ------------------------------------------------------------------
    # PyG conversion
    # ------------------------------------------------------------------

    def networkx_to_pyg(self,
                        G: nx.Graph,
                        df: pd.DataFrame,
                        include_edge_features: bool = True,
                        use_cache: bool = True) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data with rich features.

        Args:
            G: NetworkX graph
            df: Original dataframe for feature computation
            include_edge_features: Whether to include edge features
            use_cache: Use cached node features if available

        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Converting to PyTorch Geometric format...")

        x = self._compute_node_features(df, G, use_cache=use_cache)

        # Node labels
        y = torch.zeros(len(self.node_id_map), dtype=torch.long)
        has_attack_col = 'is_attack' in df.columns
        has_src_ip = 'src_ip' in df.columns
        has_dst_ip = 'dst_ip' in df.columns
        for node_id in range(len(self.node_id_map)):
            if node_id in G.nodes() and has_src_ip and has_attack_col:
                ip = self.reverse_node_map.get(node_id)
                if ip is not None:
                    attack_mask = df['is_attack'] == 1
                    src_attacks = df[(df['src_ip'] == ip) & attack_mask]
                    dst_attacks = df[(df['dst_ip'] == ip) & attack_mask] if has_dst_ip else pd.DataFrame()
                    if len(src_attacks) > 0 or len(dst_attacks) > 0:
                        y[node_id] = 1

        # Edges
        edge_list = []
        edge_attrs = []

        for src, dst, data in G.edges(data=True):
            edge_list.append([src, dst])
            if include_edge_features:
                edge_feat = [
                    float(data.get('weight', 1.0)),
                    float(data.get('total_packets', 0)),
                    float(data.get('flow_duration', 0)),
                    float(data.get('bytes_per_sec', 0)),
                    float(data.get('is_attack', 0))
                ]
                edge_attrs.append(edge_feat)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if (include_edge_features and edge_attrs) else None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None

        if self.self_loops:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(self.node_id_map)
        )

        logger.info(f"PyG Data: {data.num_nodes} nodes, {data.num_edges} edges, features {data.x.shape}")
        logger.info(f"Labels — Benign: {(y == 0).sum()}, Malicious: {(y == 1).sum()}")

        return data

    # ------------------------------------------------------------------
    # Temporal snapshots
    # ------------------------------------------------------------------

    def create_temporal_snapshots(self,
                                  df: pd.DataFrame,
                                  timestamp_col: str = 'timestamp',
                                  max_snapshots: Optional[int] = None) -> List[Data]:
        """
        Create temporal graph snapshots for time-series modeling.

        Args:
            df: DataFrame with timestamp information
            timestamp_col: Name of timestamp column
            max_snapshots: Maximum number of snapshots to create

        Returns:
            List of PyTorch Geometric Data objects
        """
        logger.info("Creating temporal graph snapshots...")

        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found. Creating single snapshot.")
            G = self.build_networkx_graph(df)
            return [self.networkx_to_pyg(G, df)]

        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

        df = df.dropna(subset=[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        min_time = df[timestamp_col].min()
        df['time_window'] = ((df[timestamp_col] - min_time).dt.total_seconds() // self.time_window).astype(int)

        unique_windows = sorted(df['time_window'].unique())

        if max_snapshots and len(unique_windows) > max_snapshots:
            step = len(unique_windows) // max_snapshots
            unique_windows = unique_windows[::step][:max_snapshots]

        logger.info(f"Creating {len(unique_windows)} temporal snapshots...")

        snapshots = []
        for window_id in unique_windows:
            window_df = df[df['time_window'] == window_id].copy()

            if len(window_df) < 10:
                continue

            try:
                # Reset node map for each window
                self.node_id_map = {}
                self.reverse_node_map = {}
                G = self.build_networkx_graph(window_df, aggregate_edges=True)
                pyg_data = self.networkx_to_pyg(G, window_df, use_cache=False)
                pyg_data.time_window = window_id
                snapshots.append(pyg_data)
            except Exception as e:
                logger.warning(f"  Failed to create snapshot for window {window_id}: {e}")
                continue

        logger.info(f"Created {len(snapshots)} valid temporal snapshots")
        return snapshots

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _update_graph_stats(self, G: nx.Graph) -> None:
        """Update internal graph statistics after construction."""
        self.graph_stats['total_nodes'] = G.number_of_nodes()
        self.graph_stats['total_edges'] = G.number_of_edges()
        self.graph_stats['avg_degree'] = (
            sum(dict(G.degree()).values()) / G.number_of_nodes()
            if G.number_of_nodes() > 0 else 0
        )
        self.graph_stats['density'] = nx.density(G)
        self.graph_stats['components'] = (
            nx.number_weakly_connected_components(G)
            if self.directed else nx.number_connected_components(G)
        )

    def compute_graph_statistics(self, G: nx.Graph) -> Dict:
        """Compute comprehensive graph statistics for analysis."""
        stats = {
            'basic': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_directed': G.is_directed()
            }
        }

        degrees = dict(G.degree())
        if degrees:
            deg_vals = list(degrees.values())
            stats['degree'] = {
                'mean': float(np.mean(deg_vals)),
                'std': float(np.std(deg_vals)),
                'min': int(min(deg_vals)),
                'max': int(max(deg_vals)),
                'median': float(np.median(deg_vals))
            }

        if G.is_directed():
            stats['connectivity'] = {
                'weakly_connected_components': nx.number_weakly_connected_components(G),
                'strongly_connected_components': nx.number_strongly_connected_components(G)
            }
        else:
            stats['connectivity'] = {
                'connected_components': nx.number_connected_components(G)
            }

        attack_nodes = [n for n, d in G.nodes(data=True) if d.get('attack_ratio', 0) > 0]
        attack_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_attack', 0) > 0]

        stats['attack'] = {
            'malicious_nodes': len(attack_nodes),
            'malicious_edges': len(attack_edges),
            'node_attack_ratio': len(attack_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'edge_attack_ratio': len(attack_edges) / G.number_of_edges() if G.number_of_edges() > 0 else 0
        }

        return stats

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def visualize_graph_sample(self,
                               G: nx.Graph,
                               output_path: str,
                               max_nodes: int = 100,
                               highlight_attacks: bool = True):
        """Visualize graph sample with attack highlighting (static PNG)."""
        try:
            import matplotlib.pyplot as plt

            if G.number_of_nodes() > max_nodes:
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
                G_sub = G.subgraph(top_nodes).copy()
            else:
                G_sub = G

            plt.figure(figsize=(16, 12))
            pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)

            if highlight_attacks:
                node_colors = []
                node_sizes = []
                for node in G_sub.nodes():
                    attack_ratio = G_sub.nodes[node].get('attack_ratio', 0)
                    if attack_ratio > 0.5:
                        node_colors.append('#FF0000')
                        node_sizes.append(300)
                    elif attack_ratio > 0:
                        node_colors.append('#FFA500')
                        node_sizes.append(200)
                    else:
                        node_colors.append('#4CAF50')
                        node_sizes.append(100)
            else:
                node_colors = '#4CAF50'
                node_sizes = 200

            nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=node_sizes,
                                   alpha=0.7, edgecolors='black', linewidths=1)

            edge_weights = [G_sub[u][v].get('weight', 1) for u, v in G_sub.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]

            nx.draw_networkx_edges(G_sub, pos, width=edge_widths, alpha=0.3,
                                   arrows=True, arrowsize=10, edge_color='gray')

            plt.title(
                f'Cyber Network Graph Sample\n{G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges',
                fontsize=16, fontweight='bold'
            )
            plt.axis('off')
            plt.tight_layout()

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Graph visualization saved to {output_path}")
            plt.close()

        except ImportError:
            logger.warning("Matplotlib not available for visualization")

    def export_interactive_html(self,
                                G: nx.Graph,
                                output_path: str,
                                max_nodes: int = 200) -> None:
        """
        Export an interactive HTML graph using pyvis (Phase 4).

        Nodes are coloured by attack involvement:
        - Red   = high attack ratio (>50 %)
        - Orange = some attacks
        - Green = benign

        Args:
            G: NetworkX graph
            output_path: Path to save the .html file
            max_nodes: Maximum nodes to include (samples by degree)
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed. Run: pip install pyvis")
            return

        if G.number_of_nodes() > max_nodes:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            G_sub = G.subgraph(top_nodes).copy()
        else:
            G_sub = G

        net = Network(height='800px', width='100%', directed=G.is_directed(),
                      bgcolor='#1a1a2e', font_color='white')
        net.set_options("""
        var options = {
          "nodes": {"borderWidth": 2, "size": 15, "font": {"size": 10}},
          "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}, "smooth": false},
          "physics": {"barnesHut": {"gravitationalConstant": -8000, "springLength": 150}}
        }
        """)

        for node, data in G_sub.nodes(data=True):
            attack_ratio = data.get('attack_ratio', 0)
            color = '#e74c3c' if attack_ratio > 0.5 else ('#e67e22' if attack_ratio > 0 else '#27ae60')
            ip_label = data.get('ip', data.get('label', str(node)))
            title = f"Node: {ip_label}<br>Attack ratio: {attack_ratio:.2%}"
            net.add_node(node, label=str(ip_label)[:20], color=color, title=title)

        for u, v, data in G_sub.edges(data=True):
            is_attack = data.get('is_attack', 0)
            color = '#e74c3c' if is_attack else '#7f8c8d'
            weight = data.get('weight', 1)
            net.add_edge(u, v, color=color, value=float(weight),
                         title=f"Flows: {weight}, Attack: {bool(is_attack)}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))
        logger.info(f"Interactive graph saved to {output_path}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_graph(self, G: nx.Graph, filepath: str):
        """Save graph to disk using pickle (gpickle)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"Graph saved to {filepath}")

    def load_graph(self, filepath: str) -> nx.Graph:
        """Load graph from disk."""
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        logger.info(f"Graph loaded from {filepath}")
        return G


if __name__ == "__main__":
    import sys

    data_path = "data/processed/cicids2017_processed.csv"

    if not Path(data_path).exists():
        logger.error(f"Processed data not found at {data_path}")
        logger.error("Please run the data loader first!")
        sys.exit(1)

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} records")

    constructor = AdvancedGraphConstructor(
        construction_method='host_based',
        time_window=300,
        directed=True,
        self_loops=False
    )

    G = constructor.build_networkx_graph(df, aggregate_edges=True, min_flows=1)

    stats = constructor.compute_graph_statistics(G)
    logger.info("Graph Statistics:")
    for category, metrics in stats.items():
        logger.info(f"\n{category.upper()}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    pyg_data = constructor.networkx_to_pyg(G, df, include_edge_features=True)

    constructor.save_graph(G, "data/graphs/cicids_network_graph.gpickle")
    torch.save(pyg_data, "data/graphs/cicids_pyg_data.pt")
    logger.info("Graph data saved successfully")

    constructor.visualize_graph_sample(
        G,
        "results/visualizations/network_graph.png",
        max_nodes=100,
        highlight_attacks=True
    )

    constructor.export_interactive_html(
        G,
        "results/visualizations/network_graph_interactive.html",
        max_nodes=200
    )

    logger.info("\nGraph construction complete!")
