# 2nd place solution(imagination-research)

- **Notebook**: https://www.kaggle.com/code/youyc22/imagination-research-2nd-place-solution/
- **Model**: https://huggingface.co/imagination-research/deepseek-14b-sft-dpo4, https://huggingface.co/imagination-research/deepseek-14b-sft-dpo2
- **Repository**: https://github.com/imagination-research/aimo2
  
Thanks to Aimo and Kaggle for hosting this great competition. We are also grateful to all the competitors during the competition and all the open source projects for sharing their great ideas. Now it's time to share our ideas and experiences with the community!

Our solution gets the 2nd place. It gets 34/50 on the public leaderboards (ranked 1st), and 31/50 (ranked 2nd) on the private leaderboard. 

## Solution Summary

This competition required optimizing both efficiency and reasoning performance. Our final solution consists of three main parts:

* Part I: **Reasoning-Oriented Training** -- *Improve the model's reasoning ability*: Stage 1 - SFT and Stage 2 - DPO with selected data.
* Part II: **Efficiency Optimization** -- *Improve inference efficiency*: Selecting a suitable inference engine, weight quantization, KV cache quantization.
* Part III: **Inference-Time Strategies** -- *Improve efficiency-reasoning performance trade-off*: Prompt design, self-consistency aggregation, sample-level/question-level early stopping, and some heuristic hyperparameter adjusting.

For local validation, we used the AIME 2025 test set (30 problems) along with the reference set (10 problems), evaluating both average sample accuracy and aggregated accuracy (via self-consistency) to obtain preliminary judgments of our trial solutions.

## Part I: Reasoning-Oriented Training

Our training scripts are based on [`Light-R1`](https://github.com/Qihoo360/Light-R1) project, we are really grateful to their work.


### Stage1: SFT

We choose [`DeepSeek-R1-Distill-Qwen-14B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) as the base model considering its great performance in mathematics, coding and reasoning.

We combine the stage2 data from [`Light-R1`](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) and training data from [`Limo`](https://huggingface.co/datasets/GAIR/LIMO) together (duplicates removed), which are both high-diffculty math problems' reasoning trajectories generated from deepseek-r1

We finetune the base model for 8 epochs on single 8×A800 machine, taking 11 hours:

<table>
  <tr>
    <td><img src="./figs/sft_accuracy.png" alt="SFT Accuracy"></td>
    <td><img src="./figs/sft_output_len.png" alt="SFT Output Length"></td>
  </tr>
</table>

The accuracy improves but the output length also improves significantly.

### Stage2: DPO

We use DPO to reduce the output length of the model

We choose the default subset of [`OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/main/data) to build our dataset

Specifically, we try to use the following three criteria to construct dpo pairs ($y_w、y_l$ means chosen response and rejected response):

* **Length ratio**:  $len(y_w) < ratio\textunderscore threshold * len(y_l)$
* **Min Length**: $len(y_w) > min\textunderscore threshold$
* **Similarity**: $sim(y_w,y_l) < si\textunderscore threshold$
  * use sentence transformer model to calculate embeddings

Applying the first two criteria, we construct the dataset dpo-1, which use to train the models we submit.

Applying the three criteria, we construct the dataset dpo-2, which we use to train another model, but its performance is similar to the model we submit.

We use [`360-LLaMA-Factory`](https://github.com/Qihoo360/360-LLaMA-Factory/tree/adfd1708b94a921637c3821bca4a6dd3d81d0387) becase they add sequence parallelism (SP) technology to support longer context training with limited memory

We use a single 8×A800 machine to train for 4 epochs on dpo-1 daatset(2k pairs), taking 40 hours

And we get two models we finally submitted:
**[`deepseek-14b-sft-dpo2`](https://huggingface.co/imagination-research/deepseek-14b-sft-dpo2)** and **[`deepseek-14b-sft-dpo4`](https://huggingface.co/imagination-research/deepseek-14b-sft-dpo4)**

ALl our training data:
**[`training data`](https://huggingface.co/datasets/imagination-research/aimo2-datasets)**

<p align="middle">
  <img src="./figs/dpo_result.png" width="80%" />
</p>

***Note: we samples 32 times (direct reasoning 16 times and code solving 16 times) for each question, the specific method can be found below***

## Part II: Efficiency Optimization

### Inference Engine

We choose [`lmdeploy`](https://github.com/InternLM/lmdeploy) as the LLM inference framework. Compared with `vllm`, the `lmdeploy` framework with the TurboMind engine can provide higher throughput and shorter model initialization time.

The first picture comes from [`here`](https://github.com/InternLM/lmdeploy)
<table>
  <tr>
    <td><img src="./figs/throughput_com.png" alt="SFT Accuracy"></td>
    <td><img src="./figs/initialization_com.png" alt="SFT Output Length"></td>
  </tr>
</table>

### Quantization

We apply 4-bit AWQ weight quantization, and 8-bit KV Cache quantization (setting the configuration `main_model.inference_cfg.quant_policy` to 8 to use 8-bit KV Cache quantization implemented by `lmdeploy`).

Our two models for submissions can be found here: [model1](https://www.kaggle.com/models/youyc22/dpsk-14b-sft-dpo-3-16-awq-tb/Keras/default), [model2](https://www.kaggle.com/models/youyc22/dpsk-14b-sft-dpo-3-13-awq-tb/Keras/default)

**Some efficiency results**:

* Online test (4xL4, batch size=15): W4KV8 decreases the time per output token by about 20% compared with W4KV16, 55% compared with FP16.
  
<p align="middle">
  <img src="./figs/output_speed.png" width="60%" />
</p>

* Local test (2xA800, batch size=32): W4KV8, W4KV16 decreases the overall latency by 40% and 20%-25% compared with FP16, respectively.

**Some reasoning performance results**:

* Local test: The average sample accuracy (not aggregated accuracy) drops by 5%~10%, compared with FP16; W4KV8 is not worse than W4KV16. W4KV4 is worse.


## Part III: Inference-Time Strategies

### Overall Inference Workflow

The inference workflow is shown in the figure below: A question is provided as the input. We first prepare two types of prompts, including the CoT prompt and the Code prompt ("Prompt Preparation Task"). Then, we let the LLM start batchify generation of multiple samples with `lmdeploy` ("LLM Generation Task"). In the mean time, we continuouly try extract the answer from the streaming output of each sample, aggregate the answers of multiple samples, and judge whether to early stop some generation:

1. We do sample-level checking upon every N yields from the iterator got by the `stream_infer(...)` call, and judge whether to early stop the generation of the corresponding sample. The Python code executor and answer extractor components are used here.
2. We do question-level checking upon the end of every sample, and judge whether to early stop the generation of all remaining samples of the current question. The answer aggregator component is used here.

Finally, we return the aggregated answer.

Note that, for each question, we adjust the speed-related hyperparameters (number of samples, sampling-level max time, question-level early-stop criterion) according to the remaining time, so that the time quota can be allocated in a more balanced way across the remaining questions when the remaining time is limited.

![Inference workflow](./figs/inference_workflow.png)

### Prompt Preparation

**Method**:

* Initially, we use 15 samples for one question, and aggregate their answers by self-consistency.
  * Note that we don't necessarily aggregate the answers of all samples, see [discussions below](#question-level-answer-aggregation--early-stopping) for more details.
  * Note that we decrease the number of samples when there is limited time left, see [discussions below](#speed-hyperparameter-adjusting) for more details.
* We use the commonly used code-based reasoning: (1) Prompt the model to provide Python code to solve the problem; (2) Extract Python code from the output, create a subprocess to execute the code; (3) Extract the answer from the execution results.
* We use two types of prompts - a CoT prompt and a Code prompt. Among 15 samples, 7 samples use the CoT prompt and 8 samples use the Code prompt:

```yaml
# CoT prompt
- system: "You are a helpful math assistant. Please reason step by step to put the answer in \\boxed{}."
  user_suffix: "\nYou excel at reasoning.\nYou must put the final answer in \\boxed{}.\nIf the final answer is greater than 1000, then take the modulo of 1000.\nThink carefully and thoroughly, avoid duplication."

# Code prompt
- system: "You are a helpful math assistant. Please provide the python code to solve the math problem and also put the final answer in \\boxed{}."
  user_suffix: "\nYou excel at coding\nYou must provide the python code, avoid redundant analysis.\nIf the final answer is greater than 1000, then take the modulo of 1000.\nThe answer must be integer.\nThere is only one answer for each question.\nImport necessary libraries."
```

**Some experiments**:

* System prompt choice: We find diversifying the system prompt doesn't help for reasoning models.
* Prompt list choice: In the local test, we find only using our Code prompt result in a consistent (across seed and models) and small improvements than using half CoT and half code prompts. However, when we submit this (only once), it doesn't help with the public submission score, thus, we don't further test this empirical choice.
* Number of samples: In the local test, we find that using 32 samples achieve better results than using 16 samples. However, due to the limited computing power on the submission platform and limited submission quota, we do not thoroughly experiment with more samples on the submission platform to find a sweet point -- we just go with 15 samples.
* How frequently the code prompt lead to code output, code error, wrong answer (32 samples, 16 CoT Prompts, 16 Code Prompts):
  * Cases where the code runs correctly but we cannot parse an integer from its output are rare and can be ignored.
  * **Before fine-tuning**, the model is more inclined to output code: on average, in only 1.9 or 3.3 out of 16 cases where a code prompt is used, the model does not output code.
  * **After our fine-tuning** with only math data, the model becomes less inclined to output code: on average, in about 11 out of 16 cases, the code prompt does not cause the model to output code. When the new model does output code, its conditional accuracy is slightly higher than the pre-fine-tuning model (45% and 55% vs. 42%).

| Model                                | Quantization | Total solving time | Avg outlen | Aggregated correct questions (/30) | Average correct samples (/32) | Code error break down (/16)                                               |
| ------------------------------------ | ------------ | ------------------ | ---------- | ---------------------------------- | ----------------------------- | ------------------------------------------------------------------------- |
| dpsk-qwen-14b                        | KV16         | 11838.22           | 9776.94    | 20.00                              | 14.63                         | No code: 1.93; Exec error: 2.97; Fail parseint: 0.13; Wrong number: 5.30  |
| dpsk-qwen-14b-awq                    | AWQ4 KV8     | 6844.75            | 10118.54   | 21.00                              | 14.40                         | No code: 3.30; Exec error: 2.53; Fail parseint: 0.33; Wrong number: 4.57  |
| dpsk-qwen-14b-finetune-v1-epoch4     | KV16         | 12971.18           | 11151.10   | 21.00                              | 18.90                         | No code: 11.00; Exec error: 0.83; Fail parseint: 0.03; Wrong number: 1.43 |
| dpsk-qwen-14b-finetune-v1-epoch4-awq | AWQ4 KV8     | 7963.94            | 11557.06   | 21.00                              | 16.80                         | No code: 11.07; Exec error: 1.30; Fail parseint: 0.03; Wrong number: 0.90 |

### Sample-level Answer Extraction & Early Stopping

**Motivation**: Usually, the reasoning model will self-doubt a lot after obtaining the answer early, even if it usually gives out the same answer finally. And in most cases, after giving the answer between `<think></think>`, the model will rewrite the solution again (at least twice). Can we reduce the wasting of tokens?

**Method**: Although we experimented with the active probing method from "Fu et al., "Efficiently Serving LLM Reasoning Programs with Certaindex, arXiv 2412", we ultimately adopted a much simpler sample-level early stopping technique to simplify our inference workflow. Specifically, once we detect either the first successfully executable code or the first answer in "\\boxed{...}", we stop the generation process for that sample.

**Some experiments**: A natural question is whether this might harm the potential to revise an initially incorrect answer into a correct one later. In our local tests, we use [`scripts/analyze_early_stop.py`](https://github.com/imagination-research/aimo2) to verify that such cases are relatively rare, as shown in the figure below.

![Early stop analysis](./figs/early_stop_analysis.jpg)

### Question-level Answer Aggregation & Early Stopping

We use the commonly used self-consistency method for answer aggregation. We use a question-level early-stop strategy as follows.

**Motivation**: The difficulty varies across problems, so we aim to avoid spending too much time on easy ones. As shown in the figure below, the output length varies considerably across samples for a single problem. This suggests that for some problems, we may obtain several correct answers early on but still need to wait for the longest sample to finish -- resulting in significant time waste (e.g., q1, q11, q16–q20, etc.).

<p align="middle">
  <img src="./figs/token_length.png" width="50%" />
</p>

**Method**: We can stop generation early for a question if sufficient certainty is achieved by examining the existing answers. Specifically, we terminate generation at the question level when a majority of the outputs are consistent, e.g., if 5 out of 7 answers agree. See the configurations in `early_stop_strategy.consistency_rules` in [`imagination_aimo2/local_eval_kaggle.py`](imagination_aimo2/local_eval_kaggle.py).

### Speed Hyperparameter Adjusting

**Motivation**: The time to solve questions of different difficulty levels varies greatly, we design a `adjust_speed` module to dynamically adjust some hyperparameters.

**Method**: As reasoning progresses, our `adjust_speed` module calculates the remaining time and number of remaining questions, and dynamically adjusts the model's sampling number and early stopping strategy accordingly.

For example, the default speed is `3(normal)`, if the system detects that the average remaining time for each question is less than 5 minutes, it automatically adjust the speed to `1(fastest)` .This means the number of samples is decreased to 10 and the maximum reasoning time for each question is also decreased. Please refer to our code for detailed implementation.

## Other methods we tried but did not work
- GRPO: we tried four training runs but did not observe significant improvement on accuracy.

- RAG: we use this [dataset](https://www.kaggle.com/datasets/artemgoncarov/math-problems-imo) as a document bank. When solving a question, we fetch 1~2 most similar questions from the bank, use them and their corresponding solutions as the few-shot demonstrations. But we did not get improvement on AIME2025 nor online submission.

- Model-based Aggregation: we give the LLM the 5 most frequent answers, and one representatitive rationale for each, ask the LLM to reason about the correct answer. But this aggregation method cannot stably achieve improvements.




