import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys

sys.path.append("/kaggle/input/lmdeploy-package")
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from transformers import set_seed
import os
import time
import warnings
import pandas as pd
import polars as pl
import kaggle_evaluation.aimo_2_inference_server
from vllm import LLM, SamplingParams
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import re
import tempfile
import subprocess
from collections import Counter
from typing import List
import time
import sys
import asyncio

sys.path.append("/kaggle/input/lmdeploy-package")
from typing import List, Tuple, Dict, Any

warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
pd.set_option("display.max_colwidth", None)
random_seed = int(time.time()) % 10000
set_seed(random_seed)
cutoff_time = time.time() + (4 * 60 + 57) * 60
max_time = time.time() + (4 * 60 + 50) * 60
global_start_time = time.time()
backend = "turbomind"
# Initialize with default values - will be adjusted dynamically based on progress
speed = 3
num_samples = 12
max_batch_size = 20
if_only_cot = False

# Define sample size mapping based on speed
SPEED_TO_SAMPLES = {
    1: 8,  # Fastest: fewer samples
    2: 10,
    3: 12,  # Default
    4: 13,
    5: 14,  # Slowest: more samples
}

llm_model_pth_14 = (
    "/kaggle/input/deepseek-r1-14b-awq-turobmind/other/default/1/deepseek-14b"
)
llm_model_pth_14_3_13 = "/kaggle/input/dpsk-14b-sft-dpo-3-13-awq-tb/keras/default/1/dpsk-14b-sft-dpo4-3-13-awq-tb"
llm_model_pth_14_3_16 = "/kaggle/input/dpsk-14b-sft-dpo-3-16-awq-tb/keras/default/1/dpsk-14b-sft-dpo2-3-16-awq-tb"

import sys

sys.path.append("/kaggle/input/lmdeploy-package")
# for only cot
thoughts = [
    "Please use chained reasoning to put the answer in \\boxed{}.",
    "Please reflect and verify while reasoning and put the answer in \\boxed{}.",
    "Solve the following problem using concise and clear reasoning by placing the answer in \\boxed{}.",
    "You are a helpful and reflective maths assistant, you reflect on each step of your reasoning.Please reason step by step to put the answer in \\boxed{}.",
    "You are the smartest maths expert in the world, please spike this question and put the answer in \\boxed{}.",
    # 'You are the cleverest maths expert in the world, please work out this question and put the answer in \\boxed{}.'
]
new_thoughts = [
    "Please use chained reasoning to put the answer in \\boxed{}.",
    "Please reflect and verify while reasoning and put the answer in \\boxed{}.",
    "Solve the following problem using concise and clear reasoning by placing the answer in \\boxed{}.",
    "You are a helpful and reflective maths assistant, you reflect on each step of your reasoning.Please reason step by step to put the answer in \\boxed{}.",
    "You are the smartest maths expert in the world, please spike this question and put the answer in \\boxed{}.",
    "Scrutinize every logical connection, validate intermediate steps, and conclude with a boxed solution: \\boxed{}.",
    "Adopt the mindset of a world-class problem-solver: methodically dissect the problem and present the answer in \\boxed{}.",
    "Deconstruct the challenge systematically, ensuring coherence at each phase, and finalize with \\boxed{}.",
    "You are a helpful maths assistant. Please reason step by step to put the answer in \\boxed{}.",
    "Please use chained reasoning to put the answer in \\boxed{}.",
    "Solve the following problem using concise and clear reasoning by placing the answer in \\boxed{}.",
    "You are a helpful maths assistant. Please reason step by step to put the answer in \\boxed{}.",
    "Drive your mathematical analysis forward with purpose and certainty, concluding in \\boxed{}.",
    "Develop your solution through focused mathematical steps and place your answer in \\boxed{}.",
    "Execute your problem-solving strategy with clarity and conviction, presenting the result in \\boxed{}.",
]
thoughts_cot = (
    # '\nDo not need to verify the answer, save time.'
    "\n You excel at reasoning."
    "\n You must put the final answer in \\boxed{} before </think>."
    "\n The final answer should modulo 1000."
    "\n Avoid duplication and improve efficiency."
)
# for code
thoughts_code = (
    # '\n Do not need to verify the answer, save time.'
    "\n You excel at coding."
    "\n Provide the python code, avoid redundant analysis"
    "\n For difficult problems, don’t think too long, just give the code as soon as possible."
    "\n The final answer should modulo 1000 and must be an integer."
    "\n There is only one answer for each question."
    "\n Import necessary libraries. "
    "\n Improve efficiency, avoid too many loop nesting"
)


class TextExtractor:
    """Class to handle various text extraction operations"""

    @staticmethod
    def extract_python_code(text: str) -> List[str]:
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [matches[-1]] if matches else []

    @staticmethod
    def process_python_code(query: str) -> Tuple[str, int]:
        query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
        current_rows = query.strip().split("\n")
        new_rows = []
        new_rows_codes = []

        for row in current_rows:
            stripped_row = row.strip()
            new_rows.append(row)
            if stripped_row and not stripped_row.startswith("#"):
                new_rows_codes.append(stripped_row)

        line_count = len(new_rows_codes)
        ans = "\n".join(new_rows)
        return ans, line_count

    @staticmethod
    def extract_boxed_text(text: str) -> int:
        pattern = r"oxed{(.*?)}"
        matches = re.findall(pattern, text)
        if not matches:
            return -1

        content = matches[-1]
        if content.isdigit():
            num = int(content)
        else:
            nums = re.findall(r"the final answer is.*?(\d+)", content)
            if not nums:
                return -1
            num = int(nums[-1])

        return num % 1000


class AnswerSelector:
    """Class to handle answer selection logic"""

    @staticmethod
    def select_answer(answers: List[int]) -> int:
        valid_answers = []
        for answer in answers:
            try:
                if int(answer) == float(answer):
                    num = int(answer)
                    if 0 < num < 1000:
                        # Lower weight for numbers less than 10
                        weight = 0.6 if num <= 20 or num % 100 == 0 else 1
                        # Add weighted frequency
                        for _ in range(int(weight * 5)):
                            valid_answers.append(num)
            except:
                pass

        if not valid_answers:
            return 49

        # Get most frequent number
        _, answer = sorted(
            [(v, k) for k, v in Counter(valid_answers).items()], reverse=True
        )[0]
        return answer % 1000


class PythonREPL:
    """Python code execution environment"""

    def __init__(self, timeout=20):
        self.timeout = timeout

    def __call__(self, query: str) -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)

            try:
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"Execution timed out after {self.timeout} seconds."

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                return True, stdout
            else:
                # Process the error message to remove the temporary file path
                error_lines = stderr.split("\n")
                cleaned_errors = []
                for line in error_lines:
                    if temp_file_path in line:
                        # Remove the path from the error line
                        line = line.replace(temp_file_path, "<temporary_file>")
                    cleaned_errors.append(line)
                cleaned_error_msg = "\n".join(cleaned_errors)
                # Include stdout in the error case
                combined_output = (
                    f"{stdout}\n{cleaned_error_msg}" if stdout else cleaned_error_msg
                )
                return False, combined_output


warnings.simplefilter("ignore")


@dataclass
class ModelConfig:
    """Configuration dataclass for model settings"""

    model_path: str
    gpu_indices: List[int]
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    cache_dir: str = None
    backend: str = "vllm"
    quant_policy: int = 8
    max_batch_size: int = 20
    use_logn_attn: bool = False
    enable_prefix_caching: bool = True
    rope_scaling_factor: float = 1.0
    max_prefill_token_num: int = 8192
    num_samples: int = 20


class LLMActor:
    """LLM interaction class"""

    def __init__(self, config: ModelConfig):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_indices))

        self._ready = True
        self.model_path = config.model_path
        self.backend = config.backend
        self.backend_config = TurbomindEngineConfig(
            tp=len(config.gpu_indices),
            download_dir=config.cache_dir,
            max_batch_size=config.max_batch_size,
            enable_prefix_caching=config.enable_prefix_caching,
            cache_max_entry_count=config.gpu_memory_utilization,
            session_len=config.max_model_len,
            max_prefill_token_num=config.max_prefill_token_num,
        )
        self.pipe = pipeline(config.model_path, self.backend_config)

    def is_ready(self):
        return self._ready

    def generate(self, texts, gen_config: GenerationConfig):
        response = self.pipe(texts, gen_config)
        return [r.text for r in response]

    async def _stop_sessions(self, pipe, start, size):
        """Helper method to stop model sessions"""
        for i in range(start, size):
            await pipe.stop_session(i + 1)

    async def _stop_one_session(self, pipe, session_id):
        """Helper method to stop a single model session"""
        await pipe.stop_session(session_id + 1)

    def _should_stop_timeout(self, valid_answers, start_time, current_speed):
        """Check if generation should time out based on speed setting"""
        solve_time = time.time()
        solved_time = solve_time - start_time

        # Base time limits
        current_end_time = 11 * 60

        # Adjust timeout criteria based on speed setting
        if current_speed <= 2:  # Fast mode - more aggressive timeouts
            current_end_time = 9 * 60 + 30
            if solved_time > 7 * 60 and len(valid_answers) >= 5:
                print("[End] Fast mode time out with 6+ answers.", flush=True)
                return True
            if solved_time > 8 * 60 and len(valid_answers) >= 4:
                print("[End] Fast mode time out with 4+ answers.", flush=True)
                return True
            if solved_time > 9 * 60 and len(valid_answers) >= 3:
                print("[End] Fast mode time out with 3+ answers.", flush=True)
                return True
        elif current_speed == 3:  # Normal mode
            current_end_time = 11 * 60
            if solved_time > 8 * 60 and len(valid_answers) >= 6:
                print("[End] Normal mode time out with 7+ answers.", flush=True)
                return True
            if solved_time > 9 * 60 and len(valid_answers) >= 5:
                print("[End] Normal mode time out with 5+ answers.", flush=True)
                return True
            if solved_time > 10 * 60 and len(valid_answers) >= 3:
                print("[End] Normal mode time out with 4+ answers.", flush=True)
                return True
        else:  # Slow mode (4-5) - more lenient timeouts
            current_end_time = 12 * 60
            if solved_time > 10 * 60 and len(valid_answers) >= 5:
                print("[End] Slow mode time out with 8+ answers.", flush=True)
                return True
            if solved_time > 11 * 60 and len(valid_answers) >= 4:
                print("[End] Slow mode time out with 6+ answers.", flush=True)
                return True

        if solved_time > current_end_time or solve_time > cutoff_time:
            print("[End] time out!", flush=True)
            return True

        return False

    def _should_stop_generation(self, valid_answers, start_time, current_speed):
        """Determine if generation should stop based on speed settings and answer patterns"""

        # Adjust stopping criteria based on speed
        min_answers = 8  # Default for speed 3

        if current_speed == 1:  # Fast modes
            min_answers = 6
        elif current_speed == 2:
            min_answers = 7
        elif current_speed == 4:
            min_answers = 9
        elif current_speed == 5:  # Slow modes
            min_answers = 10

        # Common stopping criteria across all speeds
        if len(valid_answers) <= 3 and any(
            valid_answers.count(x) >= 3 for x in valid_answers
        ):
            print(
                "[End]: An answer repeated 3 times in less than 3 valid answers.",
                flush=True,
            )
            return True
        if len(valid_answers) >= 4:
            recent_five = valid_answers[-4:]
            if any(recent_five.count(x) >= 3 for x in recent_five):
                print(
                    "[End]: An answer repeated 3 times in recent 4 valid answers.",
                    flush=True,
                )
                return True
        if len(valid_answers) <= 7 and any(
            valid_answers.count(x) >= 4 for x in valid_answers
        ):
            print(
                "[End]: An answer repeated 4 times in less than 7 valid answers.",
                flush=True,
            )
            return True
        if len(valid_answers) <= 9 and any(
            valid_answers.count(x) >= 5 for x in valid_answers
        ):
            print(
                "[End]: An answer repeated 5 times in less than 9 valid answers.",
                flush=True,
            )
            return True

        # Check for enough answers based on speed
        if len(valid_answers) >= min_answers:
            print(
                f"[End]: Collected {min_answers} answers (speed={current_speed}).",
                flush=True,
            )
            return True

        return False

    def stream_generate(self, messages, gen_config, current_speed):
        """Stream generation with early stopping based on specific criteria"""
        text_extractor = TextExtractor()
        python_repl = PythonREPL()
        cot_answers = []
        code_answers = []
        valid_answers = []
        num_messages = len(messages)
        outputs = [""] * num_messages  # Store complete output for each prompt
        token_counts = [0] * num_messages  # Store token count for each prompt
        completed_status = [
            False
        ] * num_messages  # Flag to mark if each prompt is completed
        check_token_markers = [0] * num_messages
        thinking_completed = [False] * num_messages  # Flag to mark if </think> is found
        start_time = time.time()
        session_id_start = next(self.pipe._session_id)
        print(f"Starting session ID: {session_id_start}", flush=True)

        try:
            for response in self.pipe.stream_infer(messages, gen_config):
                if all(completed_status):  # Stop if all prompts are completed
                    print("[End]: All outputs completed.", flush=True)
                    break

                # Safely access index with error handling
                try:
                    index = (
                        response.index if response is not None else 0
                    )  # Get current output index
                    if index >= len(messages):
                        print(
                            f"[Warning]: Received index {index} outside range of messages ({len(messages)})",
                            flush=True,
                        )
                        continue

                    session_id = session_id_start + index

                    if not completed_status[index]:
                        # Safely append text with None check
                        if response.text is not None:
                            outputs[
                                index
                            ] += (
                                response.text
                            )  # Append current token to corresponding prompt output
                            token_counts[index] += 1  # Increment token count

                            if self._should_stop_timeout(
                                valid_answers, start_time, current_speed
                            ):
                                break
                            # Early stopping logic
                            # Check if </think> is found in the current output
                            # Check for stopping criteria every 20 tokens
                            if token_counts[index] - check_token_markers[index] >= 20:
                                check_token_markers[index] = token_counts[index]

                                # For chain-of-thought (index % 2 == 0), check for \boxed{num}
                                if index % 2 == 0 or if_only_cot:
                                    import re

                                    # Look for \boxed{num} where num is an integer
                                    boxed_pattern = re.search(
                                        r"oxed\{(\d+)\}", outputs[index]
                                    )
                                    if boxed_pattern:
                                        asyncio.run(
                                            self._stop_one_session(
                                                self.pipe, session_id
                                            )
                                        )
                                        completed_status[index] = True
                                        current_time = time.time()
                                        time_consumed = current_time - start_time
                                        speed = (
                                            token_counts[index] / time_consumed
                                            if time_consumed > 0
                                            else 0
                                        )
                                        print(
                                            f"[Early Output] {index} completed (found boxed). Time:{time_consumed:.2f}, Token:{token_counts[index]}, Speed:{speed:.2f}",
                                            flush=True,
                                        )

                                        # Process the output
                                        self._process_cot_output(
                                            index,
                                            outputs[index],
                                            text_extractor,
                                            token_counts[index],
                                            cot_answers,
                                            valid_answers,
                                        )
                                        if self._should_stop_generation(
                                            valid_answers, start_time, current_speed
                                        ):
                                            break
                                # For code outputs (index % 2 == 1), we need to check if we've found a complete Python code block
                                elif index % 2 == 1:
                                    # Check if there's a complete Python code block
                                    if (
                                        "```python" in outputs[index]
                                        and "```"
                                        in outputs[index][
                                            outputs[index].rfind("```python") + 10 :
                                        ]
                                    ):
                                        asyncio.run(
                                            self._stop_one_session(
                                                self.pipe, session_id
                                            )
                                        )
                                        completed_status[index] = True
                                        current_time = time.time()
                                        time_consumed = current_time - start_time
                                        speed = (
                                            token_counts[index] / time_consumed
                                            if time_consumed > 0
                                            else 0
                                        )
                                        print(
                                            f"[Early Output] {index} completed (found code). Time:{time_consumed:.2f}, Token:{token_counts[index]}, Speed:{speed:.2f}",
                                            flush=True,
                                        )

                                        # Process the output
                                        self._process_code_output(
                                            index,
                                            outputs[index],
                                            text_extractor,
                                            python_repl,
                                            token_counts[index],
                                            code_answers,
                                            valid_answers,
                                        )
                                        if self._should_stop_generation(
                                            valid_answers, start_time, current_speed
                                        ):
                                            break

                        # Check if complete (based on finish_reason)
                        if (
                            response.finish_reason == "stop"
                            and not completed_status[index]
                        ):
                            completed_status[index] = True
                            current_time = time.time()
                            time_consumed = current_time - start_time
                            speed = (
                                token_counts[index] / time_consumed
                                if time_consumed > 0
                                else 0
                            )
                            print(
                                f"[Output] {index} completed normally. Time:{time_consumed:.2f}, Token:{token_counts[index]}, Speed:{speed:.2f}",
                                flush=True,
                            )

                            # Handle chain-of-thought output
                            if index % 2 == 0:
                                self._process_cot_output(
                                    index,
                                    outputs[index],
                                    text_extractor,
                                    token_counts[index],
                                    cot_answers,
                                    valid_answers,
                                )
                            # Handle code output
                            elif index % 2 == 1:
                                self._process_code_output(
                                    index,
                                    outputs[index],
                                    text_extractor,
                                    python_repl,
                                    token_counts[index],
                                    code_answers,
                                    valid_answers,
                                )
                            if self._should_stop_generation(
                                valid_answers, start_time, current_speed
                            ):
                                break
                        elif response.finish_reason is not None:
                            print(
                                f"[End]: Output {index} finished with reason: {response.finish_reason}.",
                                flush=True,
                            )
                except (AttributeError, TypeError) as e:
                    # Handle case where response or its attributes might be None
                    print(f"[Warning]: Error processing response: {e}", flush=True)
                    continue
        except Exception as e:
            print(
                f"[Error]: Exception during stream inference: {type(e).__name__} - {e}",
                flush=True,
            )
        finally:
            # Ensure we clean up even if errors occurred
            try:
                # Clean up model sessions
                asyncio.run(
                    self._stop_sessions(self.pipe, session_id_start, num_messages)
                )
            except Exception as e:
                print(f"[Error]: Failed to stop sessions: {e}", flush=True)

            # Handle any incomplete outputs
            for i in range(len(messages)):
                if not completed_status[i] and token_counts[i] > 0:
                    print(
                        f"[Warning]: Output {i} did not complete properly. Processing anyway. Token:{token_counts[i]}",
                        flush=True,
                    )
                    if i % 2 == 0:
                        self._process_cot_output(
                            i,
                            outputs[i],
                            text_extractor,
                            token_counts[i],
                            cot_answers,
                            valid_answers,
                        )
                    elif i % 2 == 1:
                        self._process_code_output(
                            i,
                            outputs[i],
                            text_extractor,
                            python_repl,
                            token_counts[i],
                            code_answers,
                            valid_answers,
                        )

        return cot_answers, code_answers, valid_answers

    def _process_cot_output(
        self, index, output, text_extractor, token_count, cot_answers, valid_answers
    ):
        """Process chain-of-thought output to extract answers"""

        boxed_answer = text_extractor.extract_boxed_text(output)

        if int(boxed_answer) == float(boxed_answer) and 0 < int(boxed_answer) < 1000:
            print(f"(Answer): extracted for Output {index}: {boxed_answer}", flush=True)
            cot_answers.append(boxed_answer)
            valid_answers.append(boxed_answer)

    def _process_code_output(
        self,
        index,
        output,
        text_extractor,
        python_repl,
        token_count,
        code_answers,
        valid_answers,
    ):
        """Process code output to extract and potentially execute Python code"""
        code_answer = 0

        # Try to extract and execute Python code
        python_code = text_extractor.extract_python_code(output)
        if python_code:
            python_code, line_count = text_extractor.process_python_code(python_code[0])
            success, output = python_repl(python_code)
            if success:
                pattern = r"(\d+)(?:\.\d+)?"  # Matches integers or decimals like 468.0
                matches = re.findall(pattern, output)
                if matches:
                    # Convert the last match to an integer by removing any decimal part
                    last_match = int(float(matches[-1]))
                    if 0 < last_match < 1000:
                        print(
                            f"<Python> result for Output {index}: {last_match}",
                            flush=True,
                        )
                        code_answer = last_match
                        code_answers.append(code_answer)
                        valid_answers.append(code_answer)
            else:
                print(f"[Error] code for Output {index}: {output}", flush=True)
        else:
            print(f"[No] code extracted for Output {index}.", flush=True)

        # extract boxed answer if present
        boxed_answer = text_extractor.extract_boxed_text(output)
        print(f"(Answer): extracted for Output {index}: {boxed_answer}", flush=True)
        if (
            0 < int(boxed_answer) < 1000
            and int(boxed_answer) == float(boxed_answer)
            and code_answer <= 0
        ):
            code_answers.append(boxed_answer)
            valid_answers.append(boxed_answer)


max_round = 1
g_score = 0
g_count = 0
total_avg_score = 0.0
total_avg_length = 0.0
total_solving_time = 0
question_times = {}
# Speed adjustment constants
TOTAL_QUESTIONS = 50
CHECK_AFTER_QUESTIONS = 30  # First check after 25 questions
CHECK_INTERVAL = 2  # Then check every 10 questions
TIME_THRESHOLDS = {
    (0, 300): 1,  # < 5:00 - very fast (speed=1)
    (300, 345): 2,  # 5:00-5:45 - fast (speed=2)
    (345, 370): 3,  # 5:45-6:10 - normal (speed=3)
    (370, 420): 4,  # 6:10-7:00 - slow (speed=4)
    (420, float("inf")): 5,  # > 7:00 - very slow (speed=5)
}
import sys

sys.path.append("/kaggle/input/lmdeploy-package")


class MathSolver:
    """Main class to handle math problem solving"""

    def __init__(self, actor, gen_config):
        self.actor = actor
        self.gen_config = gen_config
        self.text_extractor = TextExtractor()
        self.answer_selector = AnswerSelector()
        self.current_speed = speed

    def adjust_speed(self):
        """Adjust speed based on progress through questions"""
        global speed, num_samples, g_count

        # Only check at specific question counts
        if g_count >= CHECK_AFTER_QUESTIONS and g_count % CHECK_INTERVAL == 0:

            # Calculate average time per question
            avg_time_remain = (cutoff_time - time.time()) / (TOTAL_QUESTIONS - g_count)
            # avg_time_remain = (time.time()-global_start_time) / (g_count)

            # Determine new speed based on estimated time
            new_speed = 3  # Default
            for time_range, speed_value in TIME_THRESHOLDS.items():
                if time_range[0] <= avg_time_remain < time_range[1]:
                    new_speed = speed_value
                    break

            # Update speed if it changed
            if new_speed != self.current_speed:
                old_speed = self.current_speed
                self.current_speed = new_speed

                # Update sample count based on new speed
                global num_samples
                num_samples = SPEED_TO_SAMPLES[new_speed]

                print(
                    f"[SPEED ADJUSTMENT] After {g_count} questions: remaining avg time: {avg_time_remain:.2f} minutes"
                )
                print(
                    f"[SPEED ADJUSTMENT] Changed speed from {old_speed} to {new_speed}, num_samples={num_samples}"
                )

                return True

        return False

    def predict_for_question(self, question: str, id_=None, correct_answer=None) -> int:
        """Predict answer for a single question"""
        global g_score, g_count, total_solving_time
        global question_times, total_avg_score, total_avg_length

        if time.time() > cutoff_time:
            return 113

        # Adjust speed based on progress
        self.adjust_speed()

        # Start timing this question
        question_start_time = time.time()

        # Prepare questions with chain-of-thought and code prompts
        question_cot = question + thoughts_cot
        question_code = question + thoughts_code
        questions = [question_cot, question_code]

        print("correct answer:", correct_answer, flush=True)
        print(
            f"Current speed setting: {self.current_speed}, num_samples: {num_samples}",
            flush=True,
        )
        print(questions[0], flush=True)

        # Create messages for the model
        list_of_messages = [
            [
                {"role": "system", "content": new_thoughts[k % 2]},
                {"role": "user", "content": questions[k % 2]},
            ]
            for k in range(num_samples)
        ]

        # Generate and process model outputs
        cot_answers, code_answers, valid_answers = self.actor.stream_generate(
            list_of_messages, self.gen_config, self.current_speed
        )

        # Combine and select final answer
        valid_answers = cot_answers + code_answers
        selected_answer = self.answer_selector.select_answer(valid_answers)

        # Print debugging information
        print("cot answers:", cot_answers, flush=True)
        print("code answers:", code_answers, flush=True)
        print("all valid answers:", valid_answers, flush=True)
        print("selected answer:", selected_answer, flush=True)

        # Calculate and store timing information
        question_end_time = time.time()
        question_duration = question_end_time - question_start_time
        question_times[id_] = question_duration
        total_solving_time += question_duration

        # Print timing information
        print(
            f"Question {id_} solving time: {question_duration:.2f} seconds", flush=True
        )
        print(
            f"Total solving time so far: {total_solving_time:.2f} seconds", flush=True
        )

        g_count += 1

        return selected_answer


def predict(
    id_: pl.DataFrame, question: pl.DataFrame, answer: pl.DataFrame = None
) -> pl.DataFrame | pd.DataFrame:
    """Inference API function for the Kaggle competition"""
    id_ = id_.item(0)
    print(id_)
    question = question.item(0)
    prediction = math_solver.predict_for_question(question, id_, 902)
    return pl.DataFrame({"id": id_, "answer": prediction})


import sys
import os

# 获取当前环境的 PYTHONPATH
original_pythonpath = os.environ.get("PYTHONPATH", "")

# 新路径
new_path = "/kaggle/input/lmdeploy-package"
# 合并原有的 PYTHONPATH 和新路径
merged_pythonpath = (
    f"{new_path}:{original_pythonpath}" if original_pythonpath else new_path
)
os.environ["PYTHONPATH"] = merged_pythonpath

if __name__ == "__main__":

    gen_config = GenerationConfig(
        temperature=0.9,
        min_p=0.1,
        skip_special_tokens=True,
        max_new_tokens=16000,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.01,
    )

    model_config = ModelConfig(
        model_path=llm_model_pth_14_3_16,
        gpu_indices=[0, 1, 2, 3],
        gpu_memory_utilization=0.97,
        max_model_len=20000,
        backend=backend,
        quant_policy=8,
        max_batch_size=max_batch_size,
        num_samples=num_samples,
    )

    print("loading model 1...", flush=True)
    actor1 = LLMActor(model_config)
    actor1.is_ready()

    math_solver = MathSolver(actor1, gen_config)
    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(
        predict
    )

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            ("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv",)
        )
