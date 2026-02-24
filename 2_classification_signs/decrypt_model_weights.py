import os
import pickle
import logging
from utils import decrypt_weights  #

# Setup a logger to track progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatchDecrypter")


def main():

    models_dir = 'saved_models'

    # Exact model names as defined in grid_search_config.json
    ALL_MODELS = [
        'ResNet18',
        'ShuffleNetV2',
        'ConvNeXt_Atto',
        'MobileNetV3_Large',
        'EfficientNet_B0',
        'MobileViT_Small',
        'EdgeNeXt_Small',
        'EfficientFormer_L1',
        'ViT_Tiny',
        'DeiT_Tiny'
    ]


    for model_name in ALL_MODELS:
        logger.info(f"\n{'=' * 50}\nProcessing model: {model_name}\n{'=' * 50}")

        # Build the path for the specific model's subdirectory
        specific_model_dir = os.path.join(models_dir, model_name)

        # 1. Dynamically define the exact paths
        key_path = os.path.join(specific_model_dir, f'best_{model_name}_key.pkl')
        encrypted_model_path = os.path.join(specific_model_dir, f'best_{model_name}.pkl')
        decrypted_model_path = os.path.join(specific_model_dir, f'best_{model_name}_decrypted.pkl')

        # Safety check: ensure the directory exists
        if not os.path.exists(specific_model_dir):
            logger.warning(f"Directory {specific_model_dir} not found. Skipping {model_name}.")
            continue

        # Safety check: ensure both the model and the key exist
        if not os.path.exists(key_path) or not os.path.exists(encrypted_model_path):
            logger.warning(
                f"Missing files! Check that both the model and the key are inside {specific_model_dir}. Skipping {model_name}.")
            continue

        # Optional check: skip if already decrypted
        if os.path.exists(decrypted_model_path):
            logger.info(f"Model {model_name} is already decrypted. Skipping to save time.")
            continue

        try:
            # 2. Load the specific private key for this run
            logger.info(f"Loading private key from: {key_path}")
            with open(key_path, 'rb') as f:
                private_key = pickle.load(f)

            # 3. Load the encrypted model weights
            logger.info(f"Loading encrypted model from: {encrypted_model_path}")
            with open(encrypted_model_path, 'rb') as f:
                encrypted_weights = pickle.load(f)

            # 4. Decrypt the weights using function from utils.py
            logger.info(f"Starting decryption of {model_name}. This may take a few minutes...")
            decrypted_weights = decrypt_weights(private_key, encrypted_weights, logger)  #

            # 5. Save the new plaintext model in the same subdirectory
            with open(decrypted_model_path, 'wb') as f:
                pickle.dump(decrypted_weights, f)

            logger.info(f"Success! Decrypted model saved to: {decrypted_model_path}")

        except Exception as e:
            logger.error(f"An error occurred while decrypting {model_name}: {e}")

    logger.info("\nBatch decryption process finished!")


if __name__ == '__main__':
    main()