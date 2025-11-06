"""
Advanced Graph Construction for Cyber Network Analysis
Implements multiple graph construction strategies with temporal modeling

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, TemporalData
from torch_geometric.utils import to_undirected, add_self_loops
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
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
    - Host-based: Nodes represent IP addresses (hosts)
    - Flow-based: Nodes represent individual flows
    - Hierarchical: Multi-level graphs (subnet -> host -> port)
    - Temporal: Time-evolving graph snapshots
    """
    
    def __init__(self, 
                 construction_method: str = 'host_based',
                 time_window: int = 300,
                 directed: bool = True,
                 self_loops: bool = False):
        """
        Initialize graph constructor.
        
        Args:
            construction_method: 'host_based', 'flow_based', or 'hierarchical'
            time_window: Time window in seconds for temporal snapshots
            directed: Create directed graph
            self_loops: Add self-loops to nodes
        """
        self.construction_method = construction_method
        self.time_window = time_window
        self.directed = directed
        self.self_loops = self_loops
        
        # Mappings
        self.node_id_map = {}
        self.reverse_node_map = {}
        self.edge_index_cache = {}
        
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
    
    def _create_node_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        """Create mapping from IP addresses to node IDs."""
        if self.construction_method == 'host_based':
            unique_ips = pd.concat([df['src_ip'], df['dst_ip']]).unique()
            self.node_id_map = {ip: idx for idx, ip in enumerate(unique_ips)}
            self.reverse_node_map = {idx: ip for ip, idx in self.node_id_map.items()}
        
        logger.info(f"Created node mapping with {len(self.node_id_map)} unique nodes")
        return self.node_id_map
    
    def _compute_node_features(self, df: pd.DataFrame, G: nx.Graph = None) -> torch.Tensor:
        """
        Compute comprehensive node features.
        
        Features include:
        - Degree centrality (in/out for directed graphs)
        - Betweenness centrality
        - PageRank
        - Attack involvement statistics
        - Traffic volume statistics
        """
        logger.info("Computing node features...")
        
        node_features_list = []
        
        for node_id in range(len(self.node_id_map)):
            if node_id not in self.reverse_node_map:
                node_features_list.append([0.0] * 15)
                continue
            
            ip = self.reverse_node_map[node_id]
            
            # Traffic statistics as source
            src_flows = df[df['src_ip'] == ip]
            dst_flows = df[df['dst_ip'] == ip]
            
            # Basic degree features
            out_degree = len(src_flows)
            in_degree = len(dst_flows)
            total_degree = out_degree + in_degree
            
            # Attack-related features
            attack_as_src = src_flows['is_attack'].sum() if 'is_attack' in df.columns else 0
            attack_as_dst = dst_flows['is_attack'].sum() if 'is_attack' in df.columns else 0
            attack_ratio = (attack_as_src + attack_as_dst) / total_degree if total_degree > 0 else 0
            
            # Traffic volume features
            total_fwd_packets = src_flows['Total Fwd Packets'].sum() if 'Total Fwd Packets' in df.columns else 0
            total_bwd_packets = dst_flows['Total Backward Packets'].sum() if 'Total Backward Packets' in df.columns else 0
            avg_flow_duration = src_flows['Flow Duration'].mean() if 'Flow Duration' in df.columns else 0
            
            # Port diversity (indicator of scanning behavior)
            unique_dst_ports = src_flows['dst_port'].nunique() if 'dst_port' in df.columns else 0
            unique_src_ports = dst_flows['src_port'].nunique() if 'src_port' in df.columns else 0
            
            # Graph-theoretic features (if graph provided)
            if G is not None and node_id in G.nodes():
                try:
                    betweenness = nx.betweenness_centrality(G, normalized=True).get(node_id, 0)
                    closeness = nx.closeness_centrality(G).get(node_id, 0)
                    clustering = nx.clustering(G).get(node_id, 0)
                except:
                    betweenness = closeness = clustering = 0
            else:
                betweenness = closeness = clustering = 0
            
            # Compile features
            features = [
                float(out_degree),
                float(in_degree),
                float(total_degree),
                float(attack_ratio),
                float(attack_as_src),
                float(attack_as_dst),
                float(total_fwd_packets),
                float(total_bwd_packets),
                float(avg_flow_duration),
                float(unique_dst_ports),
                float(unique_src_ports),
                float(betweenness),
                float(closeness),
                float(clustering),
                float(out_degree / (in_degree + 1))  # Out/in ratio
            ]
            
            node_features_list.append(features)
        
        node_features = torch.tensor(node_features_list, dtype=torch.float)
        logger.info(f"Node features computed. Shape: {node_features.shape}")
        
        return node_features
    
    def _compute_edge_features(self, edge_df: pd.DataFrame) -> torch.Tensor:
        """
        Compute edge features from aggregated flows.
        
        Edge features include:
        - Total packets (forward + backward)
        - Total bytes transferred
        - Average flow duration
        - Number of flows
        - Attack flag
        - Protocol type
        """
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
    
    def build_networkx_graph(self, 
                            df: pd.DataFrame,
                            aggregate_edges: bool = True,
                            min_flows: int = 1) -> nx.Graph:
        """
        Build NetworkX graph with advanced aggregation.
        
        Args:
            df: DataFrame with network flow data
            aggregate_edges: Aggregate multiple flows between same hosts
            min_flows: Minimum number of flows to create an edge
            
        Returns:
            NetworkX Graph or DiGraph
        """
        logger.info(f"Building NetworkX graph using {self.construction_method} method...")
        
        # Create appropriate graph type
        G = nx.DiGraph() if self.directed else nx.Graph()
        
        # Create node mapping
        self._create_node_mapping(df)
        
        # Add nodes with initial attributes
        for ip, node_id in self.node_id_map.items():
            G.add_node(node_id, ip=ip)
        
        # Aggregate flows if requested
        if aggregate_edges:
            logger.info("Aggregating flows between host pairs...")
            
            agg_dict = {
                'Total Fwd Packets': 'sum',
                'Total Backward Packets': 'sum',
                'Flow Duration': 'mean',
                'Flow Bytes/s': 'mean',
                'is_attack': 'max',  # Edge is malicious if any flow is malicious
            }
            
            # Only aggregate columns that exist
            agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
            
            edge_data = df.groupby(['src_ip', 'dst_ip']).agg(agg_dict).reset_index()
            edge_data['num_flows'] = df.groupby(['src_ip', 'dst_ip']).size().values
            
            # Filter by minimum flows
            edge_data = edge_data[edge_data['num_flows'] >= min_flows]
            
            logger.info(f"Aggregated {len(df)} flows into {len(edge_data)} edges")
        else:
            edge_data = df
        
        # Add edges to graph
        edges_added = 0
        for _, row in edge_data.iterrows():
            src_id = self.node_id_map.get(row['src_ip'])
            dst_id = self.node_id_map.get(row['dst_ip'])
            
            if src_id is None or dst_id is None:
                continue
            
            # Compute edge weight
            weight = float(row.get('num_flows', 1))
            
            # Edge attributes
            edge_attrs = {
                'weight': weight,
                'is_attack': int(row.get('is_attack', 0)),
                'total_packets': float(
                    row.get('Total Fwd Packets', 0) + row.get('Total Backward Packets', 0)
                ),
                'flow_duration': float(row.get('Flow Duration', 0)),
                'bytes_per_sec': float(row.get('Flow Bytes/s', 0))
            }
            
            G.add_edge(src_id, dst_id, **edge_attrs)
            edges_added += 1
        
        # Update graph statistics
        self.graph_stats['total_nodes'] = G.number_of_nodes()
        self.graph_stats['total_edges'] = G.number_of_edges()
        self.graph_stats['avg_degree'] = (
            sum(dict(G.degree()).values()) / G.number_of_nodes() 
            if G.number_of_nodes() > 0 else 0
        )
        self.graph_stats['density'] = nx.density(G)
        self.graph_stats['components'] = nx.number_weakly_connected_components(G) if self.directed else nx.number_connected_components(G)
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        logger.info(f"Average degree: {self.graph_stats['avg_degree']:.2f}")
        logger.info(f"Density: {self.graph_stats['density']:.6f}")
        logger.info(f"Connected components: {self.graph_stats['components']}")
        
        return G
    
    def networkx_to_pyg(self, 
                       G: nx.Graph,
                       df: pd.DataFrame,
                       include_edge_features: bool = True) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data with rich features.
        
        Args:
            G: NetworkX graph
            df: Original dataframe for feature computation
            include_edge_features: Whether to include edge features
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Converting to PyTorch Geometric format...")
        
        # Compute node features
        x = self._compute_node_features(df, G)
        
        # Create node labels (1 if involved in attacks, 0 otherwise)
        y = torch.zeros(len(self.node_id_map), dtype=torch.long)
        for node_id in range(len(self.node_id_map)):
            if node_id in G.nodes():
                ip = self.reverse_node_map[node_id]
                src_attacks = df[(df['src_ip'] == ip) & (df['is_attack'] == 1)]
                dst_attacks = df[(df['dst_ip'] == ip) & (df['is_attack'] == 1)]
                if len(src_attacks) > 0 or len(dst_attacks) > 0:
                    y[node_id] = 1
        
        # Extract edges and edge features
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
        
        # Convert to tensors
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            if include_edge_features and edge_attrs:
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            else:
                edge_attr = None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None
        
        # Add self-loops if requested
        if self.self_loops:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(self.node_id_map)
        )
        
        # Log statistics
        logger.info(f"PyG Data created:")
        logger.info(f"  Nodes: {data.num_nodes}")
        logger.info(f"  Edges: {data.num_edges}")
        logger.info(f"  Node features: {data.x.shape}")
        if data.edge_attr is not None:
            logger.info(f"  Edge features: {data.edge_attr.shape}")
        logger.info(f"  Labels - Benign: {(y==0).sum()}, Malicious: {(y==1).sum()}")
        logger.info(f"  Attack ratio: {(y==1).sum().item() / len(y):.2%}")
        
        return data
    
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
        
        # Parse timestamps
        if df[timestamp_col].dtype == 'object':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        df = df.dropna(subset=[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Create time windows
        min_time = df[timestamp_col].min()
        df['time_window'] = ((df[timestamp_col] - min_time).dt.total_seconds() // self.time_window).astype(int)
        
        unique_windows = sorted(df['time_window'].unique())
        
        if max_snapshots and len(unique_windows) > max_snapshots:
            # Sample windows evenly
            step = len(unique_windows) // max_snapshots
            unique_windows = unique_windows[::step][:max_snapshots]
        
        logger.info(f"Creating {len(unique_windows)} temporal snapshots...")
        
        snapshots = []
        for window_id in unique_windows:
            window_df = df[df['time_window'] == window_id].copy()
            
            if len(window_df) < 10:  # Skip very small windows
                continue
            
            logger.info(f"  Window {window_id}: {len(window_df)} flows")
            
            try:
                G = self.build_networkx_graph(window_df, aggregate_edges=True)
                pyg_data = self.networkx_to_pyg(G, window_df)
                pyg_data.time_window = window_id
                snapshots.append(pyg_data)
            except Exception as e:
                logger.warning(f"  Failed to create snapshot for window {window_id}: {e}")
                continue
        
        logger.info(f"Created {len(snapshots)} valid temporal snapshots")
        return snapshots
    
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
        
        # Degree statistics
        degrees = dict(G.degree())
        if degrees:
            stats['degree'] = {
                'mean': np.mean(list(degrees.values())),
                'std': np.std(list(degrees.values())),
                'min': min(degrees.values()),
                'max': max(degrees.values()),
                'median': np.median(list(degrees.values()))
            }
        
        # Connectivity
        if G.is_directed():
            stats['connectivity'] = {
                'weakly_connected_components': nx.number_weakly_connected_components(G),
                'strongly_connected_components': nx.number_strongly_connected_components(G)
            }
        else:
            stats['connectivity'] = {
                'connected_components': nx.number_connected_components(G)
            }
        
        # Attack statistics
        attack_nodes = [n for n, d in G.nodes(data=True) if d.get('attack_ratio', 0) > 0]
        attack_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_attack', 0) > 0]
        
        stats['attack'] = {
            'malicious_nodes': len(attack_nodes),
            'malicious_edges': len(attack_edges),
            'node_attack_ratio': len(attack_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'edge_attack_ratio': len(attack_edges) / G.number_of_edges() if G.number_of_edges() > 0 else 0
        }
        
        return stats
    
    def visualize_graph_sample(self,
                               G: nx.Graph,
                               output_path: str,
                               max_nodes: int = 100,
                               highlight_attacks: bool = True):
        """Visualize graph sample with attack highlighting."""
        try:
            import matplotlib.pyplot as plt
            
            if G.number_of_nodes() > max_nodes:
                # Sample high-degree nodes
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
                G_sub = G.subgraph(top_nodes).copy()
            else:
                G_sub = G
            
            plt.figure(figsize=(16, 12))
            
            # Layout
            pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)
            
            # Node colors and sizes
            if highlight_attacks:
                node_colors = []
                node_sizes = []
                for node in G_sub.nodes():
                    attack_ratio = G_sub.nodes[node].get('attack_ratio', 0)
                    if attack_ratio > 0.5:
                        node_colors.append('#FF0000')  # Red for high attack
                        node_sizes.append(300)
                    elif attack_ratio > 0:
                        node_colors.append('#FFA500')  # Orange for some attack
                        node_sizes.append(200)
                    else:
                        node_colors.append('#4CAF50')  # Green for benign
                        node_sizes.append(100)
            else:
                node_colors = '#4CAF50'
                node_sizes = 200
            
            # Draw nodes
            nx.draw_networkx_nodes(G_sub, pos, 
                                  node_color=node_colors,
                                  node_size=node_sizes,
                                  alpha=0.7,
                                  edgecolors='black',
                                  linewidths=1)
            
            # Draw edges with varying thickness
            edge_weights = [G_sub[u][v].get('weight', 1) for u, v in G_sub.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]
            
            nx.draw_networkx_edges(G_sub, pos,
                                  width=edge_widths,
                                  alpha=0.3,
                                  arrows=True,
                                  arrowsize=10,
                                  edge_color='gray')
            
            plt.title(f'Cyber Network Graph Sample\n'
                     f'{G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges',
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Graph visualization saved to {output_path}")
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
    
    def save_graph(self, G: nx.Graph, filepath: str):
        """Save graph to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gpickle(G, filepath)
        logger.info(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath: str) -> nx.Graph:
        """Load graph from disk."""
        G = nx.read_gpickle(filepath)
        logger.info(f"Graph loaded from {filepath}")
        return G


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Load processed data
    data_path = "data/processed/cicids2017_processed.csv"
    
    if not Path(data_path).exists():
        logger.error(f"Processed data not found at {data_path}")
        logger.error("Please run the data loader first!")
        sys.exit(1)
    
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} records")
    
    # Initialize constructor
    constructor = AdvancedGraphConstructor(
        construction_method='host_based',
        time_window=300,
        directed=True,
        self_loops=False
    )
    
    # Build graph
    G = constructor.build_networkx_graph(df, aggregate_edges=True, min_flows=1)
    
    # Compute statistics
    stats = constructor.compute_graph_statistics(G)
    logger.info("Graph Statistics:")
    for category, metrics in stats.items():
        logger.info(f"\n{category.upper()}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
    
    # Convert to PyG
    pyg_data = constructor.networkx_to_pyg(G, df, include_edge_features=True)
    
    # Save outputs
    constructor.save_graph(G, "data/graphs/cicids_network_graph.gpickle")
    torch.save(pyg_data, "data/graphs/cicids_pyg_data.pt")
    logger.info("Graph data saved successfully")
    
    # Visualize sample
    constructor.visualize_graph_sample(
        G, 
        "results/visualizations/network_graph.png",
        max_nodes=100,
        highlight_attacks=True
    )
    
    logger.info("\n✅ Graph construction complete!")