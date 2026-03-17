import json
import os
import re
from tqdm import tqdm

def extract_answer(text: str):
    """
    Helper to extract the numeric value inside <answer> tags.
    """
    # Matches <answer>6000</answer> or <answer> 6000.0 </answer>
    match = re.search(r"<answer>\s*([\d\.]+)\s*</answer>", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Implements Rejection Sampling (RFT) to generate CoT reasoning paths.
    """
    # 1. Initialize your CoT model (SmolLM2-1.7B-Instruct)
    # This assumes CoTModel is already defined in your environment
    from homework.cot import CoTModel
    checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model = CoTModel(checkpoint) 
    
    # 2. Load your initial training data
    # Assuming your Dataset class from data.py is available
    from homework.data import Dataset
    train_raw = Dataset("train")
    length_dataset=len(train_raw)
    print(f"length of dataset={length_dataset}",train_raw[0])
    questions = [train_raw[i][0] for i in range(length_dataset)]
    print("set of questions:", questions)
    gold_answers=[train_raw[i][1] for i in range(length_dataset)]
    print("set of gold answers",gold_answers)
    rft_dataset = []
    
    prompts = [model.format_prompt(q) for q in questions]
    batch_results = model.batched_generate(prompts,10,0.8)
    # predicted_answers [model.parse_answer(g) for g in batch_results]

        # batch_results is a list of lists: [[Q1_rollouts], [Q2_rollouts], ...]
    for q_idx, rollouts in enumerate(batch_results):
        gold = gold_answers[q_idx]
        question = questions[q_idx]
        # predicted_answer = predicted_answer[q_idx]
            
        for completion in rollouts:
            if abs(model.parse_answer(completion) - float(gold)) < 1e-5:
                rft_dataset.append([question, gold, completion])
                break
            else:
              print("----question:",question,"correct answer:",gold,"incorrect answer:",model.parse_answer(completion),"reasoning",completion,"----")
    # 5. Save results to the specified path
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(rft_dataset, f, indent=2)
    
    print(f"RFT generation complete. Saved {len(rft_dataset)} samples to {output_json}.")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
