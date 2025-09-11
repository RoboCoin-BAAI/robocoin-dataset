import json
import logging
import re
from pathlib import Path

import requests

from robocoin_dataset.annotation.constant import (
    DS_API_KEY,
    END_FRAME_IDX,
    NEW_SUBTASK_ANNOTATION_FILE_PATH,
    START_FRAME_IDX,
    SUB_TASK,
    SUBTASK_ANNOTATION_FILE_PATH,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient


def _get_video_labels_dict(data: dict) -> dict:
    result = []
    for item in data:
        for label in item.get("videoLabels", []):
            for idx in range(len(label["ranges"])):
                result.append(  # noqa: PERF401
                    {
                        START_FRAME_IDX: label["ranges"][idx]["start"],
                        END_FRAME_IDX: label["ranges"][idx]["end"],
                        SUB_TASK: label["timelinelabels"][idx],
                    }
                )
    return result


# Step 1: 读取原始 JSON，提取所有 timelinelabels 的中文标签
def _extract_labels(data: dict) -> list[str]:
    labels_set = set()
    for item in data:
        labels_set.add(item.get(SUB_TASK))
    return sorted(labels_set)


def _detect_language(text: str) -> str:
    """
    检测文本语言
    Returns: 'zh' (中文), 'en' (英文), or 'unknown'
    """
    # 中文检测：包含中文字符
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    # 英文检测：主要包含英文字母
    if re.search(r"[a-zA-Z]", text) and not re.search(r"[\u4e00-\u9fff]", text):
        return "en"
    return "unknown"


def _process_batch_commands(command_list: list[str], ds_api_key: str) -> list[str]:
    """
    智能处理指令列表：中文翻译为英文，英文给出更好的建议

    Args:
        command_list: 指令文本列表，可以是中文或英文

    Returns:
        处理后的英文文本列表

    Raises:
        Exception: 处理失败时抛出异常
    """
    # print("正在使用DeepSeek API处理指令...")

    # 分类指令
    zh_commands = []
    en_commands = []

    for i, command in enumerate(command_list):
        lang = _detect_language(command)
        if lang == "zh":
            zh_commands.append((i, command))
        elif lang == "en":
            en_commands.append((i, command))
        else:
            # 未知语言按中文处理（或者可以根据需求调整）
            zh_commands.append((i, command))

    # 准备提示词
    prompt_parts = []
    # print(zh_commands)
    # print(en_commands)

    if zh_commands:
        zh_text = "\n".join(f"{idx + 1}. {cmd}" for idx, (orig_idx, cmd) in enumerate(zh_commands))
        # print(f"zh_text: {zh_text}")
        prompt_parts.append(f"""
            请将以下中文动作指令逐条翻译为自然的英文语句，保持语义准确：
            {zh_text}
        """)

    if en_commands:
        en_text = "\n".join(f"{idx + 1}. {cmd}" for idx, (orig_idx, cmd) in enumerate(en_commands))
        # print(f"en_text: {en_text}")
        prompt_parts.append(f"""
            请将以下英文动作指令优化为更自然、地道的英文语句：
            {en_text}
        """)

    if not prompt_parts:
        return command_list  # 如果没有需要处理的指令，直接返回

    prompt = "\n".join(prompt_parts)
    prompt += "\n\n请按顺序给出处理结果，不要编号，每行一个结果："

    # DeepSeek API配置
    api_url = "https://api.deepseek.com/v1/chat/completions"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ds_api_key}"}

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": 1024,
        "stream": False,
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)

        if response.status_code == 200:
            response_data = response.json()
            processed_text = response_data["choices"][0]["message"]["content"].strip()
            # print(f"processed_text: {processed_text}")

            # 处理结果
            result_lines = [line.strip() for line in processed_text.split("\n") if line.strip()]
            # print(f"result_lines: {result_lines}")

            # 清理可能的编号
            cleaned_results = []
            for line in result_lines:
                if ". " in line and line.split(". ")[0].isdigit():
                    cleaned_results.append(line.split(". ", 1)[-1])
                else:
                    cleaned_results.append(line)

            # print(f"cleaned_results: {cleaned_results}")

            # 验证结果数量
            expected_count = len(zh_commands) + len(en_commands)
            if len(cleaned_results) != expected_count:
                raise ValueError(
                    f"处理结果数量({len(cleaned_results)})与预期数量({expected_count})不匹配"
                )
            final_results = [""] * len(command_list)

            # 填充中文翻译结果
            for (orig_idx, _), result in zip(zh_commands, cleaned_results[: len(zh_commands)]):
                final_results[orig_idx] = result

            # 填充英文优化结果
            en_start_idx = len(zh_commands)
            for (orig_idx, _), result in zip(
                en_commands, cleaned_results[en_start_idx : en_start_idx + len(en_commands)]
            ):
                final_results[orig_idx] = result

            # 处理未知语言或未处理的指令
            for i in range(len(command_list)):
                if not final_results[i]:
                    final_results[i] = command_list[i]  # 保持原样

            for item in final_results:
                item = item.rstrip(".")
            return final_results

        raise Exception(f"API请求失败: {response.status_code} - {response.text}")

    except Exception as e:
        raise Exception(f"处理指令时出错: {str(e)}")


# Step 3: 构建映射表，并验证数量一致
def _build_mapping(zh_list: list[str], en_list: list[str]) -> dict:
    if len(zh_list) != len(en_list):
        raise ValueError("翻译数量不一致")
    return dict(zip(zh_list, en_list))


# Step 4: 重构 JSON，添加英文标签
def _transform_labels(original_data: list[dict], mapping: dict) -> list[str]:
    new_data = original_data.copy()
    for item in new_data:
        if SUB_TASK in item:
            item[SUB_TASK] = mapping.get(item[SUB_TASK], "")
    return new_data


def _modify_frame_idx(data: dict) -> dict:
    modifed_data = data.copy()
    for item in modifed_data:
        item[START_FRAME_IDX] = item[START_FRAME_IDX] - 1
        item[END_FRAME_IDX] = item[END_FRAME_IDX] - 1


class SubtaskAnnotationTaskClient(TaskClient):
    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )

    def get_task_category(self) -> str:
        return "subtask_annotation"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        # print(f"task_content: {task_content}")
        try:
            subtask_annotation_file_path = task_content.get(SUBTASK_ANNOTATION_FILE_PATH)
            ds_api_key = task_content.get(DS_API_KEY)
            if not Path(subtask_annotation_file_path).exists():
                raise FileNotFoundError(f"{subtask_annotation_file_path} not exists")
            with open(subtask_annotation_file_path) as f:
                data = json.load(f)
                video_labels_dict = _get_video_labels_dict(data)
                labels = _extract_labels(video_labels_dict)

                en_labels = _process_batch_commands(labels, ds_api_key=ds_api_key)

                mapping = _build_mapping(labels, en_labels)
                # for k, v in mapping.items():
                #     print(f"{k} -> {v}")

                transformed_labels = _transform_labels(video_labels_dict, mapping)
                # print(f"transformed_labels: {transformed_labels}")
                frame_modifed_labels = _modify_frame_idx(transformed_labels)

                subtask_annotation_file_path_new = (
                    Path(subtask_annotation_file_path).parent / NEW_SUBTASK_ANNOTATION_FILE_PATH
                )
                subtask_annotation_file_path_new.parent.mkdir(parents=True, exist_ok=True)
                with open(subtask_annotation_file_path_new, "w") as new_f:
                    json.dump(frame_modifed_labels, new_f)

        except Exception as e:
            raise RuntimeError(
                f"convert subtask annotation file {subtask_annotation_file_path} failed"
            ) from e
