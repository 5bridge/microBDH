import os
import torch
import bdh

BDH_CONFIG = bdh.MicroBDHConfig() 
MODEL_PATH = 'bdh_micro.pt'
MAX_NEW_TOKENS = 300 
TEMPERATURE = 1  
TOP_K = 20    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctx = torch.no_grad() 

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: File with model weights ‘{MODEL_PATH}’ not found.")
        print("Please run training first to generate this file.")
        exit()
        
    print(f"Loading model from file ‘{MODEL_PATH}’...")
    
    model = bdh.BDH(BDH_CONFIG)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    
    print("Model successfully loaded.")
    return model

def main():
    model = load_model()
    print("Enter your query. To exit, type ‘exit’ or ‘quit’.")

    while True:
        prompt_str = input("\n[USER] > ")
        
        if prompt_str.lower() in ['exit', 'quit']:
            break
        if not prompt_str:
            continue
            
        formatted_prompt = f"[USER]{prompt_str}[ASSISTANT]"
        
        prompt_tensor = torch.tensor(bytearray(formatted_prompt, "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
        
        # 4. Генерируем ответ
        print("--- [Model] is thinking... ---")
        with ctx:
            output_tensor = model.generate(
                idx=prompt_tensor,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K
            )

        full_response_bytes = bytes(output_tensor.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="ignore")
        try:
            model_answer = full_response_bytes.split("[ASSISTANT]")[1]
            model_answer = model_answer.split("<|endoftext|>")[0]
        except IndexError:
            model_answer = "Could not extract an answer from the generated sequence."
            
        print(model_answer.strip())


if __name__ == "__main__":
    main()