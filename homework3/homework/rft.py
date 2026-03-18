from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


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
          r=16, 
          lora_alpha=64, # 4x rank as recommended
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
          per_device_train_batch_size=32,
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
