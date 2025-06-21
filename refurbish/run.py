# import argparse
# import yaml
# from addict import Dict
# import train
# import cell_stats

# def load_config(config_path):
#     """Load the YAML configuration file and return it as an addict.Dict."""
#     with open(config_path, 'r') as f:
#         config_data = yaml.safe_load(f)
#     return Dict(config_data)

# def main():
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description="Run training with a specified config.")
#     parser.add_argument('--config', required=True, help="Path to the YAML config file")
#     parser.add_argument('--cell', help="Specific cell ID to process (optional)")
#     args = parser.parse_args()

#     # Load configuration and run training
#     config = load_config(args.config)
#     train.run_train(config, specific_cell=args.cell)
#     # cell_stats.cell_stats(config)

# if __name__ == "__main__":
#     main()

import argparse
import yaml
from addict import Dict
import plan
import os
import re

def interpolate_vars(data, context=None):
    """Recursively interpolate variables in the config."""
    if context is None:
        context = data

    if isinstance(data, dict):
        return {key: interpolate_vars(value, context) for key, value in data.items()}
    elif isinstance(data, list):
        return [interpolate_vars(item, context) for item in data]
    elif isinstance(data, str):
        # Replace ${var} with the actual value
        pattern = r'\${([^}]*)}'
        while True:
            match = re.search(pattern, data)
            if not match:
                break
            var_path = match.group(1)
            parts = var_path.split('.')
            value = context
            for part in parts:
                value = value[part]
            data = data.replace(match.group(0), str(value))
        return data
    else:
        return data

def load_config(config_path):
    """Load the YAML configuration file and return it as an addict.Dict."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # First convert to Dict for easier access
    config = Dict(config_data)
    
    # Then interpolate variables
    interpolated = interpolate_vars(config_data)
    
    # Convert back to Dict
    return Dict(interpolated)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run operations with a specified config.")
    parser.add_argument('--config', required=True, help="Path to the YAML config file")
    parser.add_argument('--plan', action='store_true', help="Process COLMAP reconstruction and create visualization")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Execute requested operations
    if args.plan:
        plan.plan(config)

if __name__ == "__main__":
    main()