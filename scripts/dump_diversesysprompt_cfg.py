import yaml

system_prompts = [
    "Please use chained reasoning to put the answer in \\boxed{}.",
    "Please reflect and verify while reasoning and put the answer in \\boxed{}.",
    (
        "Solve the following problem using concise and clear reasoning by placing the"
        " answer in \\boxed{}."
    ),
    (
        "You are a helpful and reflective maths assistant, you reflect on each step of"
        " your reasoning. Please reason step by step to put the answer in \\boxed{}."
    ),
    (
        "You are the smartest maths expert in the world, please spike this question and"
        " put the answer in \\boxed{}."
    ),
    (
        "Scrutinize every logical connection, validate intermediate steps, and conclude"
        " with a boxed solution: \boxed{}."
    ),
    (
        "Adopt the mindset of a world-class problem-solver: methodically dissect the"
        " problem and present the answer in \boxed{}."
    ),
    (
        "Deconstruct the challenge systematically, ensuring coherence at each phase,"
        " and finalize with \boxed{}."
    ),
    (
        "You are a helpful maths assistant. Please reason step by step to put the"
        " answer in \\boxed{}."
    ),
    "Please use chained reasoning to put the answer in \\boxed{}.",
    (
        "Solve the following problem using concise and clear reasoning by placing the"
        " answer in \\boxed{}."
    ),
    (
        "You are a helpful maths assistant. Please reason step by step to put the"
        " answer in \\boxed{}."
    ),
    (
        "Drive your mathematical analysis forward with purpose and certainty,"
        " concluding in \boxed{}."
    ),
    (
        "Develop your solution through focused mathematical steps and place your answer"
        " in \boxed{}."
    ),
    (
        "Execute your problem-solving strategy with clarity and conviction, presenting"
        " the result in \boxed{}."
    ),
    (
        "You are doing a mathematical competition that require solve following problems"
        " in an accurate way. Please put the final answer in \\boxed{}."
    ),
]

with open("cfgs/dpsk-qwen-14b-finetune-v1-epoch4-awq.yaml", "r") as rf:
    cfg = yaml.safe_load(rf)

suffix_1 = (
    "\nYou excel at reasoning.\nYou must put the final answer in \\boxed{}.\nIf the"
    " final answer is greater than 1000, then take the modulo of 1000.\nThink carefully"
    " and thoroughly, avoid duplication."
)
suffix_2 = (
    "\nYou excel at coding\nYou must provide the python code, avoid redundant"
    " analysis.\nIf the final answer is greater than 1000, then take the modulo of"
    " 1000.\nThe answer must be integer.\nThere is only one answer for each"
    " question.\nImport necessary libraries."
)

cfg["actor"]["prompt_list"] = []

for sys_prompt in system_prompts:
    cfg["actor"]["prompt_list"].append({"system": sys_prompt, "user_suffix": suffix_1})
    cfg["actor"]["prompt_list"].append({"system": sys_prompt, "user_suffix": suffix_2})

assert len(cfg["actor"]["prompt_list"]) == 32
with open("cfgs/dpsk-qwen-14b-finetune-v1-epoch4-awq-diversesysprompt.yaml", "w") as wf:
    yaml.safe_dump(cfg, wf)
