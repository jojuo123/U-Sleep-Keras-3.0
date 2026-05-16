import gc
import keras
from keras import backend as K


def clear_memory(model=None, session=False):
    """
    Clear CPU/GPU memory depending on the active Keras backend.

    Supports:
    - tensorflow
    - torch (PyTorch)
    - jax

    Parameters
    ----------
    model : optional
        Keras model or any object holding GPU tensors.
    """

    backend = keras.backend.backend()

    # Remove model reference
    if model is not None:
        del model

    # Force Python garbage collection
    gc.collect()

    # TensorFlow backend
    if backend == "tensorflow":
        import tensorflow as tf
        if session:
            keras.backend.clear_session()
        gc.collect()

    # PyTorch backend
    elif backend == "torch":
        import torch

        if session:
            keras.backend.clear_session()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            gc.collect()

    # JAX backend
    elif backend == "jax":
        if session:
            keras.backend.clear_session()
        gc.collect()

        try:
            import jax

            # Synchronize and release buffers
            jax.clear_caches()
        except Exception:
            pass

    else:
        raise ValueError(f"Unsupported backend: {backend}")
    


def is_model_on_gpu(model):
    """
    Returns True if at least one weight of the model is on GPU.
    Works for Keras 3 backends: TensorFlow, JAX, PyTorch.
    """
    backend_name = K.backend()
    
    # ---------------- TensorFlow ----------------
    if backend_name == "tensorflow":
        import tensorflow as tf
        for w in model.weights:
            # if "/GPU:" in w.device:
            #     return True
            return w.device
        return False
    
    # ---------------- JAX ----------------
    elif backend_name == "jax":
        import jax
        for w in model.weights:
            # if "gpu" in str(w.device()).lower():
                # return True
            return str(w.device()).lower()
        return False
    
    # ---------------- PyTorch ----------------
    elif backend_name == "torch":
        import torch
        for w in model.trainable_weights:
            return (w.name, w.value.device)
            
        return False
    
    # ---------------- Other / PlaidML / Unknown ----------------
    else:
        try:
            devices = [K.device()]
            for d in devices:
                if "GPU" in d.upper():
                    return True
            return False
        except:
            return False