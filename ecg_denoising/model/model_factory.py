"""
model_factory_fixed.py - Fixed standalone version that works when run directly.
Use this for code runners or direct execution.
"""

import os
import sys
import torch

# Add parent directory to path for config import
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import config

# Import models directly using absolute paths
try:
    from model.lstm_model import ECGDenoisingLSTM
    from model.cnn_model import ECGDenoisingCNN
    from model.rnn_model import ECGDenoisingRNN
    from model.transformer_model import ECGDenoisingTransformer
    from model.embedding_model import ECGTokenizedDenoiser
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all model files are in the same directory")
    IMPORTS_AVAILABLE = False

def create_model(model_type: str = None) -> torch.nn.Module:
    """
    Create model based on configuration.
    
    Args:
        model_type: Optional override for config.MODEL_TYPE
        
    Returns:
        torch.nn.Module: The requested model
    """
    if model_type is None:
        model_type = config.MODEL_TYPE.lower()
    
    print(f"Creating model: {model_type.upper()}")
    
    if model_type == "lstm":
        return ECGDenoisingLSTM()
    elif model_type == "cnn":
        return ECGDenoisingCNN()
    elif model_type == "rnn":
        return ECGDenoisingRNN()
    elif model_type == "transformer":
        return ECGDenoisingTransformer()
    elif model_type == "embedding":
        return ECGTokenizedDenoiser()
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                       f"Supported: lstm, cnn, rnn, transformer, embedding")

def get_model_parameters_count(model_type: str = None) -> int:
    """Get parameter count for the specified model type."""
    model = create_model(model_type)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model_type: str = None):
    """Print model summary for the specified type."""
    model = create_model(model_type)
    
    if model_type is None:
        model_type = config.MODEL_TYPE.lower()
    
    # Print common training parameters
    print(f"\n{'='*80}")
    print("COMMON TRAINING PARAMETERS (All Models)")
    print(f"{'='*80}")
    for key, value in config.COMMON_TRAINING_PARAMS.items():
        print(f"  {key:<15s}: {value}")
    print(f"{'='*80}")
    
    # Print model-specific parameters
    print(f"\n{model_type.upper()} MODEL PARAMETERS")
    print(f"{'='*80}")
    
    if model_type == "lstm":
        print(f"  Layers: {config.LSTM_LAYERS}")
        print(f"  Dropout: {config.LSTM_DROPOUT}")
        print(f"  Bidirectional: {config.BIDIRECTIONAL}")
    elif model_type == "cnn":
        print(f"  Layers: {config.CNN_LAYERS}")
        print(f"  Kernel size: {config.CNN_KERNEL_SIZE}")
        print(f"  Pool size: {config.CNN_POOL_SIZE}")
    elif model_type == "rnn":
        print(f"  Layers: {config.RNN_LAYERS}")
        print(f"  Dropout: {config.RNN_DROPOUT}")
    elif model_type == "transformer":
        print(f"  D_model: {config.TRANSFORMER_D_MODEL}")
        print(f"  Heads: {config.TRANSFORMER_NHEAD}")
        print(f"  Layers: {config.TRANSFORMER_NUM_LAYERS}")
        print(f"  Dropout: {config.TRANSFORMER_DROPOUT}")
    elif model_type == "embedding":
        print(f"  Vocab size: {config.EMBEDDING_VOCAB_SIZE}")
        print(f"  Patch size: {config.EMBEDDING_PATCH_SIZE}")
        print(f"  D_patch: {config.EMBEDDING_D_PATCH}")
        print(f"  D_model: {config.EMBEDDING_D_MODEL}")
        print(f"  Heads: {config.EMBEDDING_NHEAD}")
        print(f"  Layers: {config.EMBEDDING_NUM_LAYERS}")
        print(f"  Commitment beta: {config.EMBEDDING_COMMITMENT_BETA}")
        print(f"  Residual hidden: {config.EMBEDDING_RESIDUAL_HIDDEN}")
    
    print(f"{'='*80}")
    print(f"  Parameters: {get_model_parameters_count(model_type):,}")
    print(f"{'='*80}\n")

def test_all_models():
    """Test all model types - works when run directly."""
    print("="*80)
    print("Testing All Model Types (Fixed Version)")
    print("="*80)
    
    models_to_test = ["lstm", "cnn", "rnn", "transformer", "embedding"]
    
    for model_type in models_to_test:
        print(f"\n{'-'*60}")
        print(f"Testing {model_type.upper()} model")
        print(f"{'-'*60}")
        
        try:
            model = create_model(model_type)
            dummy = torch.randn(2, config.WINDOW_SIZE, config.N_LEADS)
            output = model(dummy)
            
            # Handle embedding model that returns (output, commit_loss)
            if isinstance(output, tuple):
                output = output[0]  # Get just the denoised signal
            
            print(f"+ {model_type.upper()} model works!")
            print(f"  Input shape:  {tuple(dummy.shape)}")
            print(f"  Output shape: {tuple(output.shape)}")
            print(f"  Parameters:   {get_model_parameters_count(model_type):,}")
            
        except Exception as e:
            print(f"- {model_type.upper()} model failed: {e}")
    
    print(f"\n{'='*80}")
    print("Model testing complete!")
    print("="*80)

if __name__ == "__main__":
    test_all_models()
