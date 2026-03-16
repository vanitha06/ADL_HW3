from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant that performs unit conversions. Be concise. "
                    "Always think step by step to ensure accuracy. "
                    "Provide your final numerical answer inside <answer> tags, like this: <answer>value</answer>."
                )
            },
            # Example 1: Length (Kilo Meters to Inches)
            {
                "role": "user", 
                "content": "Convert 2 kilometers to inches"
            },
            {
                "role": "assistant", 
                "content": (
                    "1. Identify the conversion factor:1 kilometer = 1000 meters and 1 meter = 39.3701 inches.\n"
                    "2. Perform the calculation: 2 *1000* 39.3701 = 78740.2\n"
                    "3. The answer is <answer>78740.2</answer> inches."
                )
            },
            # Example 2: Weight (Kg to Lbs)
            {
                "role": "user", 
                "content": "Can you change 5 kilograms to pounds?"
            },
            {
                "role": "assistant", 
                "content": (
                    "1. Identify the conversion factor: 1 kilogram = 2.20462 pounds.\n"
                    "2. Perform the calculation: 5 * 2.20462 = 11.0231.\n"
                    "3. <answer>11.023</answer> pounds"
                )
            },
            # Example 3: Volume (Liters to Gallons)
            {
                "role": "user", 
                "content": "How much is 10 liters in gallons?"
            },
            {
                "role": "assistant", 
                "content": (
                    "1. Identify the conversion factor: 1 liter = 0.264172 gallons.\n"
                    "2. Perform the calculation: 10 * 0.264172 = 2.64172.\n"
                    "<answer>2.642</answer> gallons"
                )
            },
            {
                "role": "user", 
                "content": "How do we translate 65 miles per hour to meters per second."
            },
            {
                "role": "assistant", 
                "content": (
                    "1. Identify conversion factors: 1 mile = 1609.34 meters; 1 hour = 3600 seconds.\n"
                    "2. Convert distance: 65 miles * 1609.34 = 104607.1 meters.\n"
                    "3. Convert time: 104607.1 meters / 3600 seconds = 29.0575.\n"
                    "<answer>29.0575</answer> m/s."
                )
            },
            {
                "role": "user", 
                "content": "Can you represent 120 square feet in square meters"
            },
            {
                "role": "assistant", 
                "content": (
                    "1. Identify conversion factor: 1 foot = 0.3048 meters.\n"
                    "2. Square the factor for area: 1 sq ft = (0.3048 * 0.3048) = 0.092903 sq m.\n"
                    "3. Perform the calculation: 120 * 0.092903 = 11.14836.\n"
                    "<answer>11.14836</answer> square meters."
                )
            },
            
            # Final User Question
            {
                "role": "user", 
                "content": question
            }
        ]
        print("question:",question)
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        return prompt



def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
