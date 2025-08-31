# saves workflow checkpoints
#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path

def clean_data(obj):
    """Convert numpy types to Python types and round floats for clean JSON"""
    _round_to = 6
    if isinstance(obj, np.floating):
        return round(float(obj), _round_to)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        return round(obj, _round_to)
    elif isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    else:
        return obj

def cached(step_name, checkpoint_key, return_fields=None, save_result=True, output_path=None):
    """Decorator for cacheable workflow steps - load from checkpoint or save result"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get output path from Config if not provided
            if output_path is None:
                # Import here to avoid circular imports
                from workflow import Config
                base_path = Config.output_path
            else:
                base_path = output_path
            
            checkpoint_file = Path(base_path) / "cache" / f"{checkpoint_key}.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                print(f"⚪️ Step: {step_name} | ✓ {step_name} already completed, skipping")
                if return_fields:
                    if isinstance(return_fields, str):
                        return checkpoint[return_fields]
                    elif isinstance(return_fields, list):
                        return tuple(checkpoint[field] for field in return_fields)
                return checkpoint
            
            # Run the function
            print(f"🔵 Step: {step_name}...")
            result = func(*args, **kwargs)
            
            # Auto-save result if enabled
            if save_result and result is not None:
                if return_fields and isinstance(return_fields, list):
                    save_data = dict(zip(return_fields, result))
                else:
                    save_data = {"result": result} if not isinstance(result, dict) else result
                
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                clean_data_obj = clean_data(save_data)
                with open(checkpoint_file, 'w') as f:
                    json.dump(clean_data_obj, f, indent=2)
            return result
        return wrapper
    return decorator
