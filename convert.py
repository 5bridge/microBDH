import torch
import os
import bdh 

CHECKPOINT_PATH = 'bdh_checkpoint.pt' 
INFERENCE_PATH = 'bdh_micro.pt'      

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Checkpoint file ‘{CHECKPOINT_PATH}’ not found.")
else:
    try:
        print(f"Loading full checkpoint from ‘{CHECKPOINT_PATH}’...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model_weights = checkpoint['model_state_dict']
            print(f"Saving model weights to ‘{INFERENCE_PATH}’...")
            torch.save(model_weights, INFERENCE_PATH)
            print(f"Done! The file ‘{INFERENCE_PATH}’ has been successfully created and is ready for inference.")
        else:
            print("Error: Key ‘model_state_dict’ not found in checkpoint.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Make sure the checkpoint file is not corrupted.")