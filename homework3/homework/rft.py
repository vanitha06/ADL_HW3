from .base_llm import BaseLLM
from .sft import test_model
from .data import Dataset


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm
def format_example(prompt: str, reason: str,answer:str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # try:
    #     # Round to 3 decimal places for consistency
    #     rounded_answer = round(float(answer), 3)
    # except (ValueError, TypeError):
    #     rounded_answer = answer
    # prompt = prompt+"wrap the answer in answer tags"
    return {"question":prompt,"answer": answer,"reasoning":reason}
def tokenize(tokenizer, question: str,reasoning:str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{reasoning}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full    
class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
  from peft import LoraConfig, get_peft_model, TaskType
  from transformers import TrainingArguments, Trainer,DataCollatorWithPadding
  import torch

  
      # If llm is None, we use the existing load() logic from base_llm
      
  llm = BaseLLM()
      # 1. Configure LoRA
      # r=16 with all-linear usually results in ~15MB adapter size
  config = LoraConfig(
          r=10, 
          lora_alpha=40, # 4x rank as recommended
          target_modules="all-linear", 
          bias="none", 
          task_type="CAUSAL_LM"
      )
  llm.model = get_peft_model(llm.model,config)
      
      # Fix for gradient checkpointing bug on GPU
  if torch.cuda.is_available():
          llm.model.enable_input_require_grads()

  train_raw = Dataset("rft")
      # 3. Define TrainingArguments
  training_args = TrainingArguments(
          output_dir=output_dir,
          logging_dir=output_dir,
          report_to="tensorboard",
          num_train_epochs=10,
          per_device_train_batch_size=200,
          learning_rate=2e-4,
          gradient_checkpointing=True,
          logging_steps=10,
          save_strategy="epoch",
          # Ensure we don't save the full model, just the adapter
          save_total_limit=1
      )
      
  train_dataset = TokenizedDataset(llm.tokenizer,train_raw,format_example)
      # 4. Initialize and Run Trainer
  trainer = Trainer(
          model=llm.model,
          args=training_args,
          train_dataset=train_dataset,
      )

  trainer.train()
      
      # 5. Save the LoRA adapter
      # The requirement is to save to homework/sft_model
  llm.model.save_pretrained("homework/rft_model")
  test_model("homework/rft_model")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
