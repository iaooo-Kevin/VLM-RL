from datasets import load_dataset, DatasetDict
import re

import re

def extractBalancedArg(text: str, macro: str) -> str | None:
    """Return the argument inside macro{...} with balanced braces."""
    needle = macro + "{"
    start = text.rfind(needle)  # use the last occurrence
    if start == -1:
        return None
    i = start + len(needle)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None

def extractFinalAnswer(solution: str) -> str:
    # 1) Balanced \boxed{...}
    boxed = extractBalancedArg(solution, "\\boxed")
    if boxed:
        return boxed.strip()

    # 2) Last LaTeX fraction (\frac or \dfrac)
    fracMatches = re.findall(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", solution)
    if fracMatches:
        num, den = fracMatches[-1]
        return f"\\frac{{{num.strip()}}}{{{den.strip()}}}"

    # 3) Last $...$ inline math
    dollarMatches = re.findall(r"\$([^$]+)\$", solution)
    if dollarMatches:
        return dollarMatches[-1].strip()

    # 4) Heuristic “is …” near the end
    tail = " ".join([ln.strip() for ln in solution.splitlines() if ln.strip()][-2:])
    m = re.search(r"(?:Thus|Therefore|Hence|So)[^\.]*?\bis\b\s*([^\.\n]+)", tail, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 5) Last numeric-ish token
    nums = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", solution)
    if nums:
        return nums[-1].strip()

    return "UNKNOWN"

def makeTaggedMessages(example):
    prob = example["problem"]
    sol = example["solution"]
    finalAns = extractFinalAnswer(sol)

    # prefer provided chat if present; otherwise build a 2-turn convo
    baseMsgs = example.get("messages")
    if not baseMsgs or len(baseMsgs) == 0:
        baseMsgs = [
            {"role": "user", "content": prob},
            {"role": "assistant", "content": sol}
        ]

    # rewrite the assistant turn to have <think>/<answer>
    newMsgs = []
    for m in baseMsgs:
        if m["role"] == "assistant":
            tagged = f"<think>{sol}</think>\n<answer>{finalAns}</answer>"
            newMsgs.append({"role": "assistant", "content": tagged})
        else:
            newMsgs.append(m)

    example["messages_tagged"] = newMsgs
    example["final_answer"] = finalAns
    return example

if __name__ == "__main__":
    ds = load_dataset("AI-MO/NuminaMath-TIR")
    ds = ds.map(makeTaggedMessages, desc="Injecting <think>/<answer> tags")
    print(ds["test"][0]["messages_tagged"])
    print(ds["test"][0]["final_answer"])
