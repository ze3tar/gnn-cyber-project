"""
CICIDS2017 Data Loader and Preprocessor
Advanced preprocessing pipeline for network intrusion detection data
Implements robust feature engineering and data quality validation

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
import os as _os
_os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CICIDS2017Loader:
    """
    Advanced data loader for CICIDS2017 dataset with comprehensive preprocessing.

    This class implements state-of-the-art preprocessing techniques including:
    - Multi-file batch loading with optional chunked reading for memory efficiency
    - Automated feature engineering for temporal and statistical patterns
    - Robust handling of class imbalance with rare-class augmentation
    - Feature normalization and outlier detection
    - Data quality validation and reporting
    - Config schema validation at startup
    """

    # Standard feature sets for network flow analysis
    FLOW_FEATURES = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Fwd IAT Total', 'Bwd IAT Total', 'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
        'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
        'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
        'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
        'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
        'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
        'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean',
        'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
        'Idle Max', 'Idle Min'
    ]

    # Attack taxonomy based on CICIDS2017 specification
    ATTACK_TAXONOMY = {
        'BENIGN': {'id': 0, 'category': 'Normal', 'severity': 0},
        'DoS Hulk': {'id': 1, 'category': 'DoS', 'severity': 3},
        'PortScan': {'id': 2, 'category': 'Reconnaissance', 'severity': 2},
        'DDoS': {'id': 3, 'category': 'DoS', 'severity': 4},
        'DoS GoldenEye': {'id': 4, 'category': 'DoS', 'severity': 3},
        'FTP-Patator': {'id': 5, 'category': 'Brute Force', 'severity': 3},
        'SSH-Patator': {'id': 6, 'category': 'Brute Force', 'severity': 3},
        'DoS slowloris': {'id': 7, 'category': 'DoS', 'severity': 3},
        'DoS Slowhttptest': {'id': 8, 'category': 'DoS', 'severity': 3},
        'Bot': {'id': 9, 'category': 'Botnet', 'severity': 4},
        'Web Attack \x96 Brute Force': {'id': 10, 'category': 'Web Attack', 'severity': 3},
        'Web Attack \x96 XSS': {'id': 11, 'category': 'Web Attack', 'severity': 3},
        'Web Attack \x96 Sql Injection': {'id': 12, 'category': 'Web Attack', 'severity': 4},
        'Infiltration': {'id': 13, 'category': 'Infiltration', 'severity': 4},
        'Heartbleed': {'id': 14, 'category': 'Vulnerability Exploit', 'severity': 5}
    }

    # Minimum samples per class below which augmentation is triggered
    AUGMENTATION_THRESHOLD = 500

    def __init__(self,
                 data_dir: str,
                 cache_dir: str = "data/cache",
                 log_dir: str = "logs"):
        """
        Initialize the CICIDS2017 data loader.

        Args:
            data_dir: Path to directory containing CICIDS2017 CSV files
            cache_dir: Directory for caching processed data
            log_dir: Directory for log files
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.df = None
        self.metadata = {
            'load_timestamp': None,
            'files_processed': [],
            'total_records': 0,
            'preprocessing_steps': [],
            'feature_stats': {}
        }

        # Feature engineering components
        self.scaler = None
        self.feature_importance = {}

        logger.info(f"Initialized CICIDS2017Loader with data_dir: {self.data_dir}")

    # ------------------------------------------------------------------
    # Config validation (Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_config(config: dict) -> None:
        """
        Validate pipeline configuration schema at startup.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: if required keys are missing or have invalid values
        """
        required_sections = ['data', 'preprocessing', 'graph', 'model', 'training', 'evaluation', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Config missing required section: '{section}'")

        valid_model_types = {'gcn', 'gat', 'sage', 'hybrid', 'temporal'}
        model_type = config.get('model', {}).get('type', '')
        if model_type not in valid_model_types:
            raise ValueError(f"Invalid model type '{model_type}'. Must be one of {valid_model_types}")

        valid_methods = {'host_based', 'flow_based', 'hierarchical'}
        graph_method = config.get('graph', {}).get('construction_method', '')
        if graph_method not in valid_methods:
            raise ValueError(f"Invalid graph construction method '{graph_method}'. Must be one of {valid_methods}")

        ratios = config.get('evaluation', {})
        total = ratios.get('train_ratio', 0) + ratios.get('val_ratio', 0) + ratios.get('test_ratio', 0)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.4f}")

        logger.info("Configuration validated successfully.")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_csv_files(self,
                       file_pattern: str = "*.csv",
                       exclude_patterns: List[str] = None,
                       chunksize: Optional[int] = None) -> pd.DataFrame:
        """
        Load and concatenate multiple CSV files with robust error handling.

        Args:
            file_pattern: Glob pattern for matching CSV files
            exclude_patterns: List of patterns to exclude (e.g., ['html', 'zip'])
            chunksize: If set, read files in chunks to reduce peak memory usage.

        Returns:
            Combined DataFrame with all loaded data
        """
        if exclude_patterns is None:
            # Exclude archive/markup formats only — do NOT exclude 'pcap' here
            # because CICIDS2017 files are named '*.pcap_ISCX.csv' and are valid CSVs.
            exclude_patterns = ['html', 'zip', '.txt']

        csv_files = [f for f in self.data_dir.glob(file_pattern)
                     if not any(pattern in f.name.lower() for pattern in exclude_patterns)]

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir} matching pattern '{file_pattern}'"
            )

        logger.info(f"Found {len(csv_files)} CSV files to process")

        dfs = []
        failed_files = []

        for csv_file in sorted(csv_files):
            logger.info(f"Loading {csv_file.name}...")
            try:
                loaded = False
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        if chunksize:
                            chunks = pd.read_csv(
                                csv_file,
                                encoding=encoding,
                                low_memory=False,
                                na_values=['', 'NA', 'NaN', 'null', 'Infinity', '-Infinity'],
                                chunksize=chunksize
                            )
                            df_temp = pd.concat(chunks, ignore_index=True)
                        else:
                            df_temp = pd.read_csv(
                                csv_file,
                                encoding=encoding,
                                low_memory=False,
                                na_values=['', 'NA', 'NaN', 'null', 'Infinity', '-Infinity']
                            )
                        loaded = True
                        break
                    except UnicodeDecodeError:
                        continue

                if not loaded:
                    raise ValueError(f"Could not decode {csv_file.name} with any supported encoding")

                if df_temp.empty:
                    logger.warning(f"  Skipping {csv_file.name}: Empty file")
                    continue

                df_temp.columns = df_temp.columns.str.strip()
                dfs.append(df_temp)
                self.metadata['files_processed'].append(csv_file.name)
                logger.info(f"  Loaded {len(df_temp):,} records from {csv_file.name}")

            except Exception as e:
                logger.error(f"  Error loading {csv_file.name}: {str(e)}")
                failed_files.append((csv_file.name, str(e)))
                continue

        if not dfs:
            raise ValueError("No data could be loaded from any CSV files")

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files")

        logger.info("Concatenating all dataframes...")
        self.df = pd.concat(dfs, ignore_index=True, sort=False)
        self.metadata['total_records'] = len(self.df)
        self.metadata['load_timestamp'] = datetime.now().isoformat()

        logger.info(f"Successfully loaded {len(self.df):,} total records")
        logger.info(f"Dataset shape: {self.df.shape}")

        return self.df

    # ------------------------------------------------------------------
    # Preprocessing steps
    # ------------------------------------------------------------------

    def standardize_column_names(self) -> None:
        """Strip, lowercase, and map column names to standard identifiers."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")

        # Strip whitespace and lowercase everything
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Map common variations to standard names
        column_mapping = {
            'source_ip': 'src_ip',
            'src_ip': 'src_ip',
            'destination_ip': 'dst_ip',
            'dst_ip': 'dst_ip',
            'source_port': 'src_port',
            'destination_port': 'dst_port',
            'protocol': 'protocol',
            'timestamp': 'timestamp',
            'label': 'label'
        }

        # Only rename columns that exist
        existing_cols = {k: v for k, v in column_mapping.items() if k in self.df.columns}
        self.df.rename(columns=existing_cols, inplace=True)

        self.metadata['preprocessing_steps'].append('standardize_column_names')
        logger.info(f"Column names standardized. Total columns: {len(self.df.columns)}")

    def remove_duplicates(self) -> None:
        """
        Remove exact duplicate rows from the dataset.

        Duplicates inflate metrics and waste compute — 11 % of CICIDS2017
        records are exact duplicates and should be dropped before training.
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)

        logger.info(f"Removed {removed:,} duplicate records ({removed / before:.1%} of total)")
        self.metadata['preprocessing_steps'].append('remove_duplicates')

    def handle_missing_values(self, strategy: str = 'advanced') -> None:
        """
        Advanced missing value imputation.

        Args:
            strategy: 'simple', 'advanced', or 'ml-based'
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        logger.info("Handling missing values...")

        # Replace infinite values
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        missing_before = self.df.isnull().sum().sum()

        if strategy == 'simple':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns

            for col in numeric_cols:
                if self.df[col].isnull().any():
                    self.df[col].fillna(self.df[col].median(), inplace=True)

            for col in categorical_cols:
                if self.df[col].isnull().any():
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        elif strategy == 'advanced':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns

            # Group-based imputation by label
            if 'label' in self.df.columns:
                for col in numeric_cols:
                    if self.df[col].isnull().any():
                        self.df[col] = self.df.groupby('label')[col].transform(
                            lambda x: x.fillna(x.median())
                        )

            # Fill remaining with global median
            for col in numeric_cols:
                if self.df[col].isnull().any():
                    self.df[col].fillna(self.df[col].median(), inplace=True)

        missing_after = self.df.isnull().sum().sum()

        logger.info(f"Missing values: {missing_before:,} → {missing_after:,}")
        self.metadata['preprocessing_steps'].append(f'handle_missing_values_{strategy}')

    def encode_labels(self) -> None:
        """Encode attack labels with taxonomic information."""
        if self.df is None:
            raise ValueError("No data loaded.")

        label_col = 'label' if 'label' in self.df.columns else None
        if label_col is None:
            logger.warning("No label column found")
            return

        logger.info("Encoding attack labels...")

        # Clean label strings
        self.df[label_col] = self.df[label_col].str.strip()

        # Map to numeric IDs
        self.df['label_id'] = self.df[label_col].map(
            lambda x: self.ATTACK_TAXONOMY.get(x, {}).get('id', -1)
        )

        # Add attack category and severity
        self.df['attack_category'] = self.df[label_col].map(
            lambda x: self.ATTACK_TAXONOMY.get(x, {}).get('category', 'Unknown')
        )
        self.df['attack_severity'] = self.df[label_col].map(
            lambda x: self.ATTACK_TAXONOMY.get(x, {}).get('severity', 0)
        )

        # Binary classification
        self.df['is_attack'] = (self.df['label_id'] != 0).astype(int)

        # Handle unknown labels
        unknown_labels = self.df[self.df['label_id'] == -1][label_col].unique()
        if len(unknown_labels) > 0:
            logger.warning(f"Found {len(unknown_labels)} unknown labels: {unknown_labels}")
            max_id = max([v['id'] for v in self.ATTACK_TAXONOMY.values()])
            for i, label in enumerate(unknown_labels):
                new_id = max_id + i + 1
                self.df.loc[self.df[label_col] == label, 'label_id'] = new_id
                self.df.loc[self.df[label_col] == label, 'attack_category'] = 'Other'
                self.df.loc[self.df[label_col] == label, 'attack_severity'] = 2

        label_dist = self.df['label_id'].value_counts().sort_index()
        logger.info(f"Label distribution:\n{label_dist}")
        logger.info(f"Attack ratio: {self.df['is_attack'].mean():.2%}")

        self.metadata['preprocessing_steps'].append('encode_labels')

    def augment_rare_classes(self, threshold: Optional[int] = None) -> None:
        """
        Oversample rare attack classes using random duplication with jitter.

        Classes with fewer than `threshold` samples are duplicated and
        Gaussian noise (std=0.01) is applied to numeric features to create
        slightly varied synthetic records.  This is a lightweight alternative
        to SMOTE that works on tabular data without external dependencies.

        Args:
            threshold: Minimum samples per class. Defaults to AUGMENTATION_THRESHOLD.
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        if 'label_id' not in self.df.columns:
            logger.warning("encode_labels() must be called before augment_rare_classes()")
            return

        threshold = threshold or self.AUGMENTATION_THRESHOLD
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['label_id', 'is_attack', 'attack_severity']]

        augmented_frames = [self.df]
        total_added = 0

        for label_id, group in self.df.groupby('label_id'):
            count = len(group)
            if count >= threshold or label_id == 0:  # Never augment BENIGN
                continue

            needed = threshold - count
            synthetic = group.sample(n=needed, replace=True, random_state=42).copy()

            # Add Gaussian jitter to numeric features
            noise = np.random.normal(0, 0.01, size=(len(synthetic), len(numeric_cols)))
            synthetic[numeric_cols] += noise

            augmented_frames.append(synthetic)
            total_added += needed
            logger.info(f"  Augmented class {label_id}: {count} → {count + needed} samples")

        if total_added > 0:
            self.df = pd.concat(augmented_frames, ignore_index=True)
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info(f"Augmentation complete: added {total_added:,} synthetic samples")
        else:
            logger.info("No classes below threshold — augmentation not needed")

        self.metadata['preprocessing_steps'].append('augment_rare_classes')

    def engineer_features(self) -> None:
        """
        Advanced feature engineering for improved model performance.
        Creates temporal, statistical, and behavioral features.
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        logger.info("Engineering advanced features...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in
                        ['label_id', 'is_attack', 'attack_severity']]

        # 1. Ratio features
        if 'Total Fwd Packets' in self.df.columns and 'Total Backward Packets' in self.df.columns:
            self.df['fwd_bwd_packet_ratio'] = (
                self.df['Total Fwd Packets'] /
                (self.df['Total Backward Packets'] + 1)
            )

        if 'Total Length of Fwd Packets' in self.df.columns and 'Total Length of Bwd Packets' in self.df.columns:
            self.df['fwd_bwd_length_ratio'] = (
                self.df['Total Length of Fwd Packets'] /
                (self.df['Total Length of Bwd Packets'] + 1)
            )

        # 2. Statistical features
        packet_features = [c for c in self.df.columns if 'Packet Length' in c]
        if len(packet_features) >= 3:
            mean_col = [c for c in packet_features if 'Mean' in c]
            std_col = [c for c in packet_features if 'Std' in c]
            if mean_col and std_col:
                self.df['packet_length_cv'] = (
                    self.df[std_col[0]] / (self.df[mean_col[0]] + 1)
                )

        # 3. Flag-based features
        flag_cols = [c for c in self.df.columns if 'Flag' in c]
        if flag_cols:
            self.df['total_flags'] = self.df[flag_cols].sum(axis=1)
            self.df['flag_diversity'] = (self.df[flag_cols] > 0).sum(axis=1)

        # 4. Temporal features
        if 'Flow Duration' in self.df.columns:
            self.df['log_flow_duration'] = np.log1p(self.df['Flow Duration'])

        # 5. Protocol-based features
        if 'protocol' in self.df.columns:
            protocol_dummies = pd.get_dummies(self.df['protocol'], prefix='proto')
            self.df = pd.concat([self.df, protocol_dummies], axis=1)

        logger.info(f"Feature engineering complete. Total features: {len(self.df.columns)}")
        self.metadata['preprocessing_steps'].append('engineer_features')

    def normalize_features(self, method: str = 'robust') -> None:
        """
        Normalize numeric features.

        Args:
            method: 'robust' (default), 'standard', or 'minmax'
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv_files() first.")

        # Select numeric columns excluding labels
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['label_id', 'is_attack', 'attack_severity']]

        if not numeric_cols:
            logger.warning("No numeric columns found for normalization")
            return

        logger.info(f"Normalizing {len(numeric_cols)} numeric features using {method} scaler...")

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.scaler = scaler
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        self.metadata['preprocessing_steps'].append(f'normalize_features_{method}')
        logger.info("Feature normalization complete.")

    def detect_and_handle_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> None:
        """
        Detect and handle outliers using statistical methods.

        Args:
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: Threshold for outlier detection
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        logger.info(f"Detecting outliers using {method} method...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in
                        ['label_id', 'is_attack', 'attack_severity', 'src_port', 'dst_port']]

        outlier_counts = {}

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = z_scores > threshold

            outlier_counts[col] = outliers.sum()

            # Cap outliers instead of removing (preserve data)
            if outliers.any():
                self.df.loc[outliers, col] = self.df[col].quantile(0.99)

        total_outliers = sum(outlier_counts.values())
        logger.info(f"Capped {total_outliers:,} outlier values across {len(outlier_counts)} features")
        self.metadata['preprocessing_steps'].append(f'handle_outliers_{method}')

    # ------------------------------------------------------------------
    # Graph feature extraction (Phase 1 fix: was incorrectly nested inside
    # detect_and_handle_outliers — now a proper class method)
    # ------------------------------------------------------------------

    def extract_graph_features(self) -> pd.DataFrame:
        """
        Extract the subset of features needed for graph construction.

        Returns:
            DataFrame containing IP columns, port/protocol columns, label
            columns, and up to 30 numeric feature columns.
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        logger.info("Extracting graph-optimized features...")

        # Find IP columns — handle both raw ('Source IP') and post-standardization
        # ('src_ip') column names to be robust against calling order.
        src_ip_col = None
        dst_ip_col = None

        for col in self.df.columns:
            cl = col.lower()
            # Match 'src_ip', 'source_ip', or any col containing 'src'+'ip' / 'source'+'ip'
            if cl in ('src_ip', 'source_ip') or \
               ('src' in cl and 'ip' in cl) or \
               ('source' in cl and 'ip' in cl):
                src_ip_col = col
            elif cl in ('dst_ip', 'destination_ip') or \
                 ('dst' in cl and 'ip' in cl) or \
                 ('destination' in cl and 'ip' in cl):
                dst_ip_col = col

        if src_ip_col is None:
            logger.warning(
                "Source IP column not found in dataset. "
                "Generating synthetic src_ip from row index for graph construction. "
                "This produces a valid graph topology but node IDs are not real hosts."
            )
            self.df['src_ip'] = (self.df.index % 254 + 1).astype(str).radd('10.0.0.')
            src_ip_col = 'src_ip'

        if dst_ip_col is None:
            logger.warning(
                "Destination IP column not found in dataset. "
                "Generating synthetic dst_ip from label_id for graph construction."
            )
            if 'label_id' in self.df.columns:
                self.df['dst_ip'] = (self.df['label_id'] % 254 + 1).astype(str).radd('10.0.1.')
            else:
                self.df['dst_ip'] = '10.0.1.1'
            dst_ip_col = 'dst_ip'

        # Build essential column list
        essential_cols = []
        if src_ip_col:
            essential_cols.append(src_ip_col)
        if dst_ip_col:
            essential_cols.append(dst_ip_col)

        for col in ['src_port', 'dst_port', 'protocol', 'label_id', 'is_attack',
                    'attack_category', 'timestamp']:
            if col in self.df.columns:
                essential_cols.append(col)

        # Add up to 30 numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:30]:
            if col not in essential_cols and col not in ['label_id', 'is_attack']:
                essential_cols.append(col)

        feature_df = self.df[essential_cols].copy()

        # Normalise IP column names so downstream code can rely on 'src_ip' / 'dst_ip'
        rename_map = {}
        if src_ip_col and src_ip_col != 'src_ip':
            rename_map[src_ip_col] = 'src_ip'
        if dst_ip_col and dst_ip_col != 'dst_ip':
            rename_map[dst_ip_col] = 'dst_ip'
        if rename_map:
            feature_df.rename(columns=rename_map, inplace=True)

        logger.info(f"Extracted {len(feature_df.columns)} graph features")
        logger.info(f"Feature columns: {list(feature_df.columns)}")

        return feature_df

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report."""
        if self.df is None:
            raise ValueError("No data loaded.")

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'missing_values': int(self.df.isnull().sum().sum()),
            'duplicate_records': int(self.df.duplicated().sum()),
            'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / 1024 ** 2),
            'label_distribution': (
                self.df['label_id'].value_counts().to_dict()
                if 'label_id' in self.df.columns else {}
            ),
            'attack_ratio': float(self.df['is_attack'].mean()) if 'is_attack' in self.df.columns else 0.0,
            'unique_src_ips': int(self.df['src_ip'].nunique()) if 'src_ip' in self.df.columns else 0,
            'unique_dst_ips': int(self.df['dst_ip'].nunique()) if 'dst_ip' in self.df.columns else 0,
            'preprocessing_steps': self.metadata['preprocessing_steps']
        }

        return report

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess_pipeline(self,
                             sample_size: Optional[int] = None,
                             normalize: bool = True,
                             handle_outliers: bool = True,
                             engineer_features: bool = True,
                             deduplicate: bool = True,
                             augment_rare: bool = False,
                             chunksize: Optional[int] = None) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline with all best practices.

        Args:
            sample_size: Optional sample size for testing
            normalize: Whether to normalize features
            handle_outliers: Whether to handle outliers
            engineer_features: Whether to create engineered features
            deduplicate: Whether to remove duplicate records (Phase 2)
            augment_rare: Whether to oversample rare attack classes (Phase 5)
            chunksize: Chunk size for memory-efficient CSV loading (Phase 2)

        Returns:
            Preprocessed DataFrame ready for graph construction
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive preprocessing pipeline")
        logger.info("=" * 60)

        # Step 1: Load data
        self.load_csv_files(chunksize=chunksize)

        # Step 2: Sample if requested
        if sample_size and sample_size < len(self.df):
            logger.info(f"Sampling {sample_size:,} records for processing...")
            if 'Label' in self.df.columns:
                self.df = self.df.groupby('Label', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // self.df['Label'].nunique()))
                ).reset_index(drop=True)
            else:
                self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # Step 3: Standardize columns
        self.standardize_column_names()

        # Step 4: Handle missing values
        self.handle_missing_values(strategy='advanced')

        # Step 5: Remove duplicates (Phase 2)
        if deduplicate:
            self.remove_duplicates()

        # Step 6: Encode labels
        self.encode_labels()

        # Step 7: Feature engineering
        if engineer_features:
            self.engineer_features()

        # Step 8: Handle outliers
        if handle_outliers:
            self.detect_and_handle_outliers(method='iqr', threshold=3.0)

        # Step 9: Normalize features
        if normalize:
            self.normalize_features(method='robust')

        # Step 10: Augment rare classes (Phase 5)
        if augment_rare:
            self.augment_rare_classes()

        # Step 11: Extract graph features
        feature_df = self.extract_graph_features()

        # Step 12: Generate quality report
        quality_report = self.get_data_quality_report()

        logger.info("=" * 60)
        logger.info("Preprocessing pipeline complete!")
        logger.info(f"Final dataset shape: {feature_df.shape}")
        quality_score = 100 - (quality_report['missing_values'] / quality_report['total_records'] * 100)
        logger.info(f"Quality score: {quality_score:.2f}%")
        logger.info("=" * 60)

        return feature_df

    def save_processed_data(self, output_path: str, save_metadata: bool = True) -> None:
        """
        Save the processed dataset and optionally a quality report.

        Args:
            output_path: Path to save the processed CSV file.
            save_metadata: Whether to save a JSON quality report alongside the data.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        feature_df = self.extract_graph_features()
        feature_df.to_csv(output_path, index=False)
        print(f"[OK] Saved processed data -> {output_path}")

        if save_metadata:
            quality_report = self.get_data_quality_report()

            def default_serializer(o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                elif isinstance(o, (np.floating,)):
                    return float(o)
                elif isinstance(o, np.ndarray):
                    return o.tolist()
                return str(o)

            meta_file = os.path.splitext(output_path)[0] + "_quality_report.json"
            with open(meta_file, "w") as f:
                json.dump(quality_report, f, indent=2, default=default_serializer)
            print(f"[OK] Saved quality report -> {meta_file}")


if __name__ == "__main__":
    import sys

    DATA_DIR = "data/raw/CICIDS2017"
    OUTPUT_PATH = "data/processed/cicids2017_processed.csv"
    SAMPLE_SIZE = 50000  # Set to None for full dataset

    try:
        loader = CICIDS2017Loader(data_dir=DATA_DIR)

        df = loader.preprocess_pipeline(
            sample_size=SAMPLE_SIZE,
            normalize=True,
            handle_outliers=True,
            engineer_features=True,
            deduplicate=True,
            augment_rare=False,
        )

        loader.save_processed_data(OUTPUT_PATH, save_metadata=True)

        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        quality_report = loader.get_data_quality_report()
        for key, value in quality_report.items():
            if key not in ['label_distribution', 'preprocessing_steps']:
                print(f"{key}: {value}")

        print("\nPreprocessing complete! Ready for graph construction.")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)
