import re

def makeFormatReward(type: str = "answer"):
    """
    Returns a reward function that checks if the completion has the required format
    with the specified tag (e.g., 'answer', 'verdict').
    Defaults to checking for <answer>...</answer> if no type is provided.
    """
    otherPattern = f"<{type}.*?</{type}>"
    pattern = rf"<think>.*?</think>\s*{otherPattern}"

    def rewardFn(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
        completionContents = [
            completion[0]["content"] for completion in completions
        ]
        matches = [
            re.fullmatch(pattern, content, re.DOTALL) for content in completionContents
        ]
        return [1.0 if match else 0.0 for match in matches]

    return rewardFn