import os
import re
import sys
import glob
import json
import yaml

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sn
from imagination_aimo2.local_eval import PythonREPL, AnswerExtractor

answer_extractor = AnswerExtractor()
python_executor = PythonREPL()


def read_file(filename, raw=False):
    if raw:
        with open(filename, "r") as rf:
            data = rf.read()
        return data

    if filename.endswith(".yaml"):
        with open(filename, "r") as rf:
            data = yaml.safe_load(rf)
    elif filename.endswith(".json"):
        with open(filename, "r") as rf:
            data = json.load(rf)
    else:
        raise
    return data


NO_VALUE = 0
WRONG_VALUE = 1
CORRECT = 2
# EXEC_ERROR = 3
# OTHER_ERROR = 4

IS_CODE_ANSWER = "code"
IS_COT_ANSWER = "cot"

item_to_cmapind = {
    (NO_VALUE, None): 0,
    (WRONG_VALUE, IS_COT_ANSWER): 1,  # dark green
    (CORRECT, IS_COT_ANSWER): 2,  # light green
    (WRONG_VALUE, IS_CODE_ANSWER): 3,  # dark blue
    (CORRECT, IS_CODE_ANSWER): 4,  # light blue
}
cmap = ["#000000", "#124f34", "#38e899", "#132147", "#3c6be8"]


def _pad_value(list_of_list, pad_value):
    num_answers_list = [len(ans_status_list) for ans_status_list in list_of_list]
    max_ans_num = max(num_answers_list)
    return [
        ans_status_list + [pad_value] * (max_ans_num - len(ans_status_list))
        for ans_status_list in list_of_list
    ], num_answers_list


def plot_answer_ratio(answer_stats_for_exp, save_path):
    """Create a stack bar plot."""
    # red: first correct -> final wrong. green: first wrong -> final correct
    fig, ax = plt.subplots()
    colors = ["#000000", "#444444", "#AAAAAA", "#cf1313", "#13cf45", "#EEEEEE"]
    sample_one_answer_q, fc_lc_q, fw_lw_q, fc_lw_q, fw_lc_q, sample_no_answer_q = zip(
        *answer_stats_for_exp
    )

    num_questions = len(sample_one_answer_q)
    xs = range(0, num_questions)
    ax.bar(xs, sample_one_answer_q, color=colors[0])
    bottom = np.array(sample_one_answer_q)

    for data, color in zip(
        [fc_lc_q, fw_lw_q, fc_lw_q, fw_lc_q, sample_no_answer_q], colors[1:]
    ):
        ax.bar(xs, data, color=color, bottom=bottom)
        bottom = bottom + np.array(data)

    ax.set_xticks(range(0, num_questions))
    ax.set_xticklabels([f"q{ind}" for ind in range(0, num_questions)], fontsize=6)

    plt.savefig(save_path)
    plt.close()


def get_answers_status_for_question(output_dir, ques_ind, correct_answer):
    answers_status_for_question = []
    earlystop_lengths_for_question = []
    samples = read_file(
        os.path.join(output_dir, "outputs_per_question", f"{ques_ind}.json")
    )

    # save python code and execution output for easy checking
    code_extraction_dir = os.path.join(
        output_dir, "outputs_per_question", "code_extraction", f"{ques_ind}"
    )
    os.makedirs(code_extraction_dir, exist_ok=True)

    for sample_ind, sample in enumerate(samples):
        cot_answers = [
            (
                match.span(1),
                answer_extractor.canonicalize_number(match.group(1)),
            )
            for match in list(re.finditer(r"oxed{(.*?)}", sample))
        ]
        # cot_answers_status_for_sample = [OTHER_ERROR if ans is None else int(ans ==  taa    correct_answer) + 1 for ans in cot_answers]
        cot_answers_status_for_sample = [
            (ans_item[0], int(ans_item[1] == correct_answer) + 1, IS_COT_ANSWER)
            for ans_item in cot_answers
            if ans_item[1] is not None
        ]

        code_answers_status_for_sample = []
        for code_ind, code_match in enumerate(
            re.finditer(r"```python\s*(.*?)\s*```", sample, re.DOTALL)
        ):
            python_code = code_match.group(1)
            exec_success, exec_output = python_executor(python_code)

            # Save the python code and exec output
            python_code_file = os.path.join(
                code_extraction_dir, f"sample{sample_ind}-code{code_ind}.py"
            )
            exec_output_file = os.path.join(
                code_extraction_dir, f"sample{sample_ind}-code{code_ind}.out"
            )
            with open(python_code_file, "w") as wf:
                wf.write(
                    f"# Matched token index: {code_match.start(1)} -"
                    f" {code_match.end(1)}\n"
                    + python_code
                )
            with open(exec_output_file, "w") as wf:
                wf.write(exec_output)

            if exec_success:
                pattern = r"(\d+)(?:\.\d+)?"  # Matches integers or decimals like 468.0
                matches = re.findall(pattern, exec_output)
                if matches:
                    answer = answer_extractor.canonicalize_number(matches[-1])
                    # if answer is None:
                    #     code_answers_status_for_sample.append(OTHER_ERROR)
                    code_answers_status_for_sample.append(
                        (
                            code_match.span(1),
                            int(answer == correct_answer) + 1,
                            IS_CODE_ANSWER,
                        )
                    )
            #     else:
            #         code_answers_status_for_sample.append(OTHER_ERROR)
            # else:
            #     code_answers_status_for_sample.append(EXEC_ERROR)
        answers_status_for_sample = sorted(
            cot_answers_status_for_sample + code_answers_status_for_sample
        )

        # Calculate how much earlier (in string length) the early-stop version get the answer
        # I don't want to call the tokenizer now, let's just calculate the string length instead of the token length!
        # print the string position of the first answer match's ending, and the total length
        if answers_status_for_sample:
            print(answers_status_for_sample[0][0][1], len(sample))
            earlystop_lengths_for_question.append(
                (answers_status_for_sample[0][0][1], len(sample))
            )
        else:
            earlystop_lengths_for_question.append((len(sample), len(sample)))
        answers_status_for_sample = [
            (item[0][0], item[1], item[2]) for item in answers_status_for_sample
        ]
        answers_status_for_question.append(answers_status_for_sample)

    return answers_status_for_question, earlystop_lengths_for_question


def save_correct_heatmap(output_dirs, plot_every_question=True, reuse_cache=True):
    for output_dir in output_dirs:
        print(f"Handling {output_dir} ...")
        cache_dir = os.path.join(output_dir, "outputs_per_question", "cache_plot")
        os.makedirs(cache_dir, exist_ok=True)

        num_questions = len(
            glob.glob(os.path.join(output_dir, "outputs_per_question", "*.json"))
        )
        results = read_file(os.path.join(output_dir, "results.json"))
        answers_status_for_exp = []
        answer_stats_for_exp = []
        for ques_ind in range(num_questions):
            correct_answer = results[ques_ind]["correct_answer"]

            # Some code run for a long time. Caching results for quicker plotting for multiple times
            cache_file_for_question = os.path.join(cache_dir, f"{ques_ind}.json")
            if reuse_cache and os.path.exists(cache_file_for_question):
                loaded = read_file(cache_file_for_question)
                if isinstance(loaded, dict):
                    answers_status_for_question, earlystop_lengths_for_question = (
                        loaded["answers_status"],
                        loaded["earlystop_lengths"],
                    )
                else:  # list, for back compatability of previous cache
                    answers_status_for_question = loaded
            else:
                answers_status_for_question, earlystop_lengths_for_question = (
                    get_answers_status_for_question(
                        output_dir, ques_ind, correct_answer
                    )
                )
                with open(cache_file_for_question, "w") as wf:
                    json.dump(
                        {
                            "answers_status": answers_status_for_question,
                            "earlystop_lengths": earlystop_lengths_for_question,
                        },
                        wf,
                    )

            num_samples = len(answers_status_for_question)

            # Plot answer statusheatmap for this question; record number of valid answers
            plot_answers_status_for_question, _ = _pad_value(
                answers_status_for_question, (None, NO_VALUE, None)
            )

            if plot_every_question:
                print(f"Plotting {output_dir}, #{ques_ind} question ...")
                plt.figure()
                ax = sn.heatmap(
                    [
                        [item_to_cmapind[tuple(item[1:])] for item in list_]
                        for list_ in plot_answers_status_for_question
                    ],
                    cmap=cmap,
                )

                ax.set_yticks(0.5 + np.arange(0, num_samples))
                ax.tick_params(axis="both", which="minor", labelsize=8)
                ax.set_yticklabels(
                    [str(ind) for ind in range(0, num_samples)], fontsize=6
                )

                ax.set_xticks(
                    0.5 + np.arange(0, len(plot_answers_status_for_question[0]))
                )
                ax.set_xticklabels(
                    [
                        str(num)
                        for num in range(
                            1, len(plot_answers_status_for_question[0]) + 1
                        )
                    ]
                )

                c_bar = ax.collections[0].colorbar
                c_bar.set_ticks([0, 1, 2, 3, 4])
                c_bar.set_ticklabels(
                    ["", "CoT wrong", "CoT correct", "Code wrong", "Code correct"]
                )
                save_path = os.path.join(
                    output_dir, "outputs_per_question", f"{ques_ind}_answer_status.pdf"
                )
                ax.get_figure().savefig(save_path)
                plt.close()

            # stat the number of no answer or one answer
            # for sample that have more than one answer: stat the number of the first answer to be correct; the ratio of the final answer to be correct; the ratio of first correct and final wrong; the ratio of first wrong and final correct
            # the average ratio of all answers to be correct
            sample_no_answer = 0
            sample_one_answer = 0
            fc_lc, fw_lw, fc_lw, fw_lc = 0, 0, 0, 0
            for answers_status_for_sample in answers_status_for_question:
                if len(answers_status_for_sample) == 0:
                    sample_no_answer += 1
                elif len(answers_status_for_sample) == 1:
                    sample_one_answer += 1
                else:
                    fc_lc += int(
                        answers_status_for_sample[0][1] == CORRECT
                        and answers_status_for_sample[-1][1] == CORRECT
                    )
                    fw_lw += int(
                        answers_status_for_sample[0][1] == WRONG_VALUE
                        and answers_status_for_sample[-1][1] == WRONG_VALUE
                    )
                    fc_lw += int(
                        answers_status_for_sample[0][1] == CORRECT
                        and answers_status_for_sample[-1][1] == WRONG_VALUE
                    )
                    fw_lc += int(
                        answers_status_for_sample[0][1] == WRONG_VALUE
                        and answers_status_for_sample[-1][1] == CORRECT
                    )
            answer_stats_for_exp.append(
                [sample_one_answer, fc_lc, fw_lw, fc_lw, fw_lc, sample_no_answer]
            )

        save_path = os.path.join(output_dir, "answer_refine_vis.pdf")
        print(f"Start plotting answer ratio to {save_path} ...")
        plot_answer_ratio(answer_stats_for_exp, save_path)


if __name__ == "__main__":
    save_correct_heatmap(sys.argv[1:])
