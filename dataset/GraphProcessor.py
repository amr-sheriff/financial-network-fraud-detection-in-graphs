import argparse
import logging
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='preprocessedData')
    parser.add_argument('--output-dir', type=str, default='GraphProcessedData')
    parser.add_argument('--transactions', type=str, default='full_transaction.csv')
    parser.add_argument('--identity', type=str, default='full_identity.csv')
    parser.add_argument('--id-cols', type=str, default='')
    parser.add_argument('--cat-cols', type=str, default='')
    parser.add_argument('--num-cols', type=str, default='')
    parser.add_argument('--setting', type=str, choices=['transductive', 'inductive'],
                        default='transductive',
                        help='Training setting: transductive (single graph) or inductive (separate graphs)')
    parser.add_argument('--normalize', action='store_true',
                        help='Whether to normalize numerical features')
    parser.add_argument('--val-transactions', type=str, default=None,
                        help='Validation transactions file for inductive setting')
    parser.add_argument('--val-identity', type=str, default=None,
                        help='Validation identity file for inductive setting')
    return parser.parse_args()

def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger

def load_data(args):
    """
    Load data based on training setting (transductive/inductive)
    """
    if args.setting == 'transductive':
        transaction_df = pd.read_csv(os.path.join(args.data_dir, args.transactions))
        identity_df = pd.read_csv(os.path.join(args.data_dir, args.identity))

        logging.info("Shape of transaction data is {}".format(transaction_df.shape))
        logging.info("Shape of identity data is {}".format(identity_df.shape))
        logging.info("# Tagged transactions: {}".format(
            len(transaction_df) - transaction_df.isFraud.isnull().sum()))

        get_fraud_frac = lambda series: 100 * sum(series)/len(series)
        logging.info("Percent fraud for all transactions: {:.2f}%".format(
            get_fraud_frac(transaction_df.isFraud)))

        return transaction_df, identity_df

    else:
        if not (args.val_transactions and args.val_identity):
            raise ValueError("Validation data files must be provided for inductive setting")

        train_df = pd.read_csv(os.path.join(args.data_dir, args.transactions))
        train_identity_df = pd.read_csv(os.path.join(args.data_dir, args.identity))

        val_df = pd.read_csv(os.path.join(args.data_dir, args.val_transactions))
        val_identity_df = pd.read_csv(os.path.join(args.data_dir, args.val_identity))

        logging.info("Training data shapes - Transactions: {}, Identity: {}".format(
            train_df.shape, train_identity_df.shape))
        logging.info("Validation data shapes - Transactions: {}, Identity: {}".format(
            val_df.shape, val_identity_df.shape))

        get_fraud_frac = lambda series: 100 * sum(series)/len(series)
        logging.info("Percent fraud - Training: {:.2f}%, Validation: {:.2f}%".format(
            get_fraud_frac(train_df.isFraud), get_fraud_frac(val_df.isFraud)))

        return train_df, val_df, train_identity_df, val_identity_df

def fit_preprocessors(df, num_cols, cat_cols, normalize=True):
    """
    Fit preprocessors on data
    """
    preprocessors = {}

    if normalize and num_cols:
        num_cols = num_cols.split(",") if isinstance(num_cols, str) else num_cols
        scaler = StandardScaler()
        scaler.fit(df[num_cols])
        preprocessors['num_scaler'] = scaler
        logging.info(f"Fitted StandardScaler on columns: {num_cols}")

    if cat_cols:
        cat_cols = cat_cols.split(",") if isinstance(cat_cols, str) else cat_cols
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(df[cat_cols])
        preprocessors['cat_encoder'] = encoder
        logging.info(f"Fitted OneHotEncoder on columns: {cat_cols}")

    return preprocessors

def transform_features(df, num_cols, cat_cols, preprocessors):
    """
    Transform features using fitted preprocessors
    """
    transformed_features = []

    if 'num_scaler' in preprocessors:
        num_cols = num_cols.split(",") if isinstance(num_cols, str) else num_cols
        num_features = preprocessors['num_scaler'].transform(df[num_cols])
        num_features = pd.DataFrame(
            num_features,
            columns=num_cols,
            index=df.index
        )
        transformed_features.append(num_features)
        logging.info(f"Transformed numerical features, shape: {num_features.shape}")

    if 'cat_encoder' in preprocessors:
        cat_cols = cat_cols.split(",") if isinstance(cat_cols, str) else cat_cols
        cat_features = preprocessors['cat_encoder'].transform(df[cat_cols])
        cat_features = pd.DataFrame(
            cat_features,
            columns=preprocessors['cat_encoder'].get_feature_names_out(cat_cols),
            index=df.index
        )
        transformed_features.append(cat_features)
        logging.info(f"Transformed categorical features, shape: {cat_features.shape}")

    if transformed_features:
        features = pd.concat(transformed_features, axis=1)
        logging.info(f"Combined feature shape: {features.shape}")
        return features
    else:
        return pd.DataFrame(index=df.index)

def get_relations_and_edgelist(transactions_df, identity_df, transactions_id_cols, output_dir, prefix=''):
    """
    Generate relation files for heterogeneous graph construction
    """
    logging.info("Generating relation files...")

    edge_types = [col for col in (transactions_id_cols.split(",") + list(identity_df.columns))
                  if col != 'TransactionID']
    logging.info(f"Found the following distinct relation types: {edge_types}")

    id_cols = ['TransactionID'] + transactions_id_cols.split(",")
    full_identity_df = transactions_df[id_cols].merge(identity_df, on='TransactionID', how='left')
    logging.info(f"Shape of merged identity data: {full_identity_df.shape}")

    edges = {}
    for etype in edge_types:
        edgelist = full_identity_df[['TransactionID', etype]].dropna()
        output_file = os.path.join(output_dir, f'{prefix}relation_{etype}_edgelist.csv')
        edgelist.to_csv(output_file, index=False, header=True)
        edges[etype] = edgelist
        logging.info(f"Wrote edgelist to: {output_file}")
        logging.info(f"Number of edges for relation {etype}: {len(edgelist)}")

    return edges

def save_preprocessors(preprocessors, output_dir):
    """
    Save fitted preprocessors
    """
    for name, preprocessor in preprocessors.items():
        output_file = os.path.join(output_dir, f'{name}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(preprocessor, f)
        logging.info(f"Saved {name} to: {output_file}")

def main():
    args = parse_args()
    logger = get_logger(__name__)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.setting == 'transductive':
        transactions_df, identity_df = load_data(args)

        preprocessors = fit_preprocessors(
            transactions_df, args.num_cols, args.cat_cols, args.normalize)

        features = transform_features(
            transactions_df, args.num_cols, args.cat_cols, preprocessors)

        features.to_csv(os.path.join(args.output_dir, 'features.csv'), index=False)
        transactions_df[['TransactionID', 'isFraud']].to_csv(
            os.path.join(args.output_dir, 'labels.csv'), index=False)

        _ = get_relations_and_edgelist(
            transactions_df, identity_df, args.id_cols, args.output_dir)

        save_preprocessors(preprocessors, args.output_dir)

    else:
        train_df, val_df, train_identity_df, val_identity_df = load_data(args)

        preprocessors = fit_preprocessors(
            train_df, args.num_cols, args.cat_cols, args.normalize)

        train_features = transform_features(
            train_df, args.num_cols, args.cat_cols, preprocessors)

        val_features = transform_features(
            val_df, args.num_cols, args.cat_cols, preprocessors)

        train_features.to_csv(os.path.join(args.output_dir, 'train_features.csv'), index=False)
        train_df[['TransactionID', 'isFraud']].to_csv(
            os.path.join(args.output_dir, 'train_labels.csv'), index=False)

        val_features.to_csv(os.path.join(args.output_dir, 'val_features.csv'), index=False)
        val_df[['TransactionID', 'isFraud']].to_csv(
            os.path.join(args.output_dir, 'val_labels.csv'), index=False)

        _ = get_relations_and_edgelist(
            train_df, train_identity_df, args.id_cols, args.output_dir, prefix='train_')
        _ = get_relations_and_edgelist(
            val_df, val_identity_df, args.id_cols, args.output_dir, prefix='val_')

        save_preprocessors(preprocessors, args.output_dir)

    logging.info("Processing completed successfully")

if __name__ == '__main__':
    main()
