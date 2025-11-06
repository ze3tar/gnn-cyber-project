"""
CICIDS2017 Data Loader
Loads and preprocesses the CICIDS2017 intrusion detection dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CICIDS2017Loader:
    """
    Loader for CICIDS2017 dataset with preprocessing capabilities
    """
    
    def __init__(self, data_dir: str = "data/raw/CICIDS2017"):
        """
        Initialize the data loader
        
        Args:
            data_dir: Path to directory containing CICIDS2017 CSV files
        """
        self.data_dir = Path(data_dir)
        self.df = None
        
        # Common column name mappings (CICIDS2017 has some inconsistent naming)
        self.column_mapping = {
            ' Source IP': 'src_ip',
            ' Destination IP': 'dst_ip',
            ' Source Port': 'src_port',
            ' Destination Port': 'dst_port',
            ' Protocol': 'protocol',
            ' Timestamp': 'timestamp',
            ' Flow Duration': 'flow_duration',
            ' Total Fwd Packets': 'fwd_packets',
            ' Total Backward Packets': 'bwd_packets',
            'Total Length of Fwd Packets': 'fwd_packet_length',
            ' Total Length of Bwd Packets': 'bwd_packet_length',
            ' Flow Bytes/s': 'flow_bytes_per_sec',
            ' Flow Packets/s': 'flow_packets_per_sec',
            ' Label': 'label'
        }
        
        # Attack type mappings
        self.attack_mapping = {
            'BENIGN': 0,
            'DoS Hulk': 1,
            'PortScan': 2,
            'DDoS': 3,
            'DoS GoldenEye': 4,
            'FTP-Patator': 5,
            'SSH-Patator': 6,
            'DoS slowloris': 7,
            'DoS Slowhttptest': 8,
            'Bot': 9,
            'Web Attack – Brute Force': 10,
            'Web Attack – XSS': 11,
            'Web Attack – Sql Injection': 12,
            'Infiltration': 13,
            'Heartbleed': 14
        }
        
    def load_csv_files(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load all CSV files from the data directory
        
        Args:
            file_pattern: Pattern to match CSV files
            
        Returns:
            Combined DataFrame
        """
        csv_files = list(self.data_dir.glob(file_pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for csv_file in csv_files:
            logger.info(f"Loading {csv_file.name}...")
            try:
                df_temp = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
                dfs.append(df_temp)
                logger.info(f"  Loaded {len(df_temp)} records from {csv_file.name}")
            except Exception as e:
                logger.error(f"  Error loading {csv_file.name}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No data could be loaded from CSV files")
        
        self.df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total records loaded: {len(self.df)}")
        
        return self.df
    
    def clean_column_names(self) -> None:
        """Clean and standardize column names"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        # Strip whitespace from column names
        self.df.columns = self.df.columns.str.strip()
        
        # Apply column mapping where available
        rename_dict = {k.strip(): v for k, v in self.column_mapping.items() 
                      if k.strip() in self.df.columns}
        self.df.rename(columns=rename_dict, inplace=True)
        
        logger.info(f"Cleaned column names. Columns: {list(self.df.columns[:10])}...")
    
    def handle_missing_values(self) -> None:
        """Handle missing and infinite values"""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Replace infinite values with NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Count missing values
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found {missing_counts.sum()} missing values")
            
            # Fill numeric columns with median
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
    
    def encode_labels(self) -> None:
        """Encode attack labels to numeric values"""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        label_col = 'label' if 'label' in self.df.columns else 'Label'
        
        if label_col not in self.df.columns:
            logger.warning("Label column not found in dataset")
            return
        
        # Map labels to numeric values
        self.df['label_encoded'] = self.df[label_col].map(self.attack_mapping)
        
        # Handle unmapped labels
        unmapped = self.df[self.df['label_encoded'].isnull()][label_col].unique()
        if len(unmapped) > 0:
            logger.warning(f"Unmapped labels found: {unmapped}")
            # Assign new IDs to unmapped labels
            max_id = max(self.attack_mapping.values())
            for i, label in enumerate(unmapped):
                self.attack_mapping[label] = max_id + i + 1
                self.df.loc[self.df[label_col] == label, 'label_encoded'] = max_id + i + 1
        
        # Create binary label (benign=0, attack=1)
        self.df['is_attack'] = (self.df['label_encoded'] != 0).astype(int)
        
        logger.info(f"Label distribution:\n{self.df['label_encoded'].value_counts()}")
        logger.info(f"Binary distribution - Benign: {(self.df['is_attack']==0).sum()}, "
                   f"Attack: {(self.df['is_attack']==1).sum()}")
    
    def extract_flow_features(self) -> pd.DataFrame:
        """
        Extract relevant features for graph construction
        
        Returns:
            DataFrame with selected features
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Define key features for graph nodes/edges
        feature_cols = []
        
        # Try to find common feature columns (with flexible naming)
        possible_features = [
            ('src_ip', ['src_ip', 'Source IP', ' Source IP']),
            ('dst_ip', ['dst_ip', 'Destination IP', ' Destination IP']),
            ('src_port', ['src_port', 'Source Port', ' Source Port']),
            ('dst_port', ['dst_port', 'Destination Port', ' Destination Port']),
            ('protocol', ['protocol', 'Protocol', ' Protocol']),
            ('flow_duration', ['flow_duration', 'Flow Duration', ' Flow Duration']),
            ('fwd_packets', ['fwd_packets', 'Total Fwd Packets', ' Total Fwd Packets']),
            ('bwd_packets', ['bwd_packets', 'Total Backward Packets', ' Total Backward Packets']),
            ('flow_bytes_per_sec', ['flow_bytes_per_sec', 'Flow Bytes/s', ' Flow Bytes/s']),
            ('label_encoded', ['label_encoded']),
            ('is_attack', ['is_attack'])
        ]
        
        selected_features = {}
        for feature_name, possible_cols in possible_features:
            for col in possible_cols:
                if col in self.df.columns:
                    selected_features[feature_name] = col
                    break
        
        if not selected_features:
            logger.warning("Could not find standard features, using first 10 columns")
            return self.df.iloc[:, :10]
        
        feature_df = self.df[[v for v in selected_features.values()]].copy()
        feature_df.columns = list(selected_features.keys())
        
        logger.info(f"Extracted {len(feature_df.columns)} features")
        return feature_df
    
    def preprocess_pipeline(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline
        
        Args:
            sample_size: Optional number of records to sample (for testing)
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        self.load_csv_files()
        
        # Sample if requested
        if sample_size and sample_size < len(self.df):
            logger.info(f"Sampling {sample_size} records...")
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Clean and process
        self.clean_column_names()
        self.handle_missing_values()
        self.encode_labels()
        
        # Extract features
        feature_df = self.extract_flow_features()
        
        logger.info("Preprocessing complete!")
        logger.info(f"Final dataset shape: {feature_df.shape}")
        
        return feature_df
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        stats = {
            'total_records': len(self.df),
            'unique_src_ips': self.df['src_ip'].nunique() if 'src_ip' in self.df.columns else 0,
            'unique_dst_ips': self.df['dst_ip'].nunique() if 'dst_ip' in self.df.columns else 0,
            'attack_ratio': (self.df['is_attack'].sum() / len(self.df)) if 'is_attack' in self.df.columns else 0,
            'label_distribution': self.df['label_encoded'].value_counts().to_dict() if 'label_encoded' in self.df.columns else {}
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = CICIDS2017Loader(data_dir="data/raw/CICIDS2017")
    
    # Run preprocessing (use sample_size for testing)
    df = loader.preprocess_pipeline(sample_size=10000)
    
    # Print statistics
    stats = loader.get_statistics()
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save processed data
    output_path = Path("data/processed/cicids2017_processed.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")