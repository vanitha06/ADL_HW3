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
                "content": "You are a helpful assistant that performs unit conversions. Be concise and show your reasoning step-by-step."
            },
            {
                "role": "user", 
                "content": "Convert 2 meters to inches."
            },
            {
                "role": "assistant", 
                "content": "1 meter is 100 cm, and 1 inch is 2.54 cm. 2 meters = 200 cm. 200 / 2.54 = 78.74. The answer is 78.74 inches."
            },
            {
                "role": "user", 
                "content": "Convert 10 feet to meters."
            },
            {
                "role": "assistant", 
                "content": "1 foot is 0.3048 meters. 10 feet * 0.3048 = 3.048. The answer is 3.048 meters."
            },
            {
                "role": "user", 
                "content": "Convert 5 kilograms to pounds."
            },
            {
                "role": "assistant", 
                "content": "1 kilogram is approximately 2.20462 pounds. 5 * 2.20462 = 11.0231. The answer is 11.0231 pounds."
},

                        {
                "role": "user", 
                "content": question
            }
        ]

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
