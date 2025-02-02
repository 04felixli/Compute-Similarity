import os
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv
import heapq
import Levenshtein

# Load environment variables from .env file
load_dotenv()

TEMPLATE_COLUMN_NAME = os.getenv("TEMPLATE_COLUMN_NAME")
LOG_COLUMN_NAME = os.getenv("LOG_COLUMN_NAME")
INPUT_FOLDER = os.getenv("INPUT_FOLDER")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
N_VALUE = int(os.getenv("N_VALUE"))
K_VALUE = int(os.getenv("K_VALUE"))


def main():
    process_all_input_files()


# Function to process all input files in the INPUT_FOLDER
def process_all_input_files():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".xlsx"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(
                OUTPUT_FOLDER, filename.split(".")[0] + "_output.xlsx"
            )
            process_single_input_file(input_path, output_path)


# Function to process a single input file and generate output
def process_single_input_file(input_path, output_path):
    res = (
        []
    )  # stores a list of (similar template pair, list of similar logs between template pair)

    templates = remove_duplicate_templates(input_path, TEMPLATE_COLUMN_NAME)

    if len(templates) == 1:
        logs = get_log_cluster(templates[0], input_path)
        res.append((None, None, find_most_dissimilar_log_pair(logs), None))
    else:
        n_most_similar_templates = find_N_most_similar_templates(templates)

        for pair in n_most_similar_templates:
            temp1_logs = get_log_cluster(pair[1], input_path)
            temp2_logs = get_log_cluster(pair[2], input_path)

            res.append(
                (
                    pair,
                    find_K_most_similar_logs_between_similar_log_templates(
                        temp1_logs, temp2_logs
                    ),
                    find_most_dissimilar_log_pair(temp1_logs),
                    find_most_dissimilar_log_pair(temp2_logs),
                )
            )

    generate_output_file(res, output_path)


def generate_output_file(
    all_logs: List[
        Tuple[
            Tuple[float, str, str] | None,  # Similar template pair
            List[Tuple[float, str, str]] | None,  # List of similar log messages
            Tuple[float, str, str],  # Dissimilar log message for template 1
            Tuple[float, str, str] | None,  # Dissimilar log message for template 2
        ]
    ],
    output_path: str,
) -> None:
    data = {
        "Template Similarity Index": [],
        "Template 1": [],
        "Template 2": [],
    }

    # Dynamically add log pairs
    for (
        template_pair,
        similar_logs,
        dissimilar_log_pair_1,
        dissimilar_log_pair_2,
    ) in all_logs:
        if template_pair is not None:
            data["Template Similarity Index"].append(template_pair[0])
            data["Template 1"].append(template_pair[1])
            data["Template 2"].append(template_pair[2])
        else:
            data["Template Similarity Index"].append(None)
            data["Template 1"].append(None)
            data["Template 2"].append(None)

        if similar_logs is not None:
            for j, log in enumerate(similar_logs):
                index_key = f"Log Pair {j+1} Similarity Index"
                log1_key = f"Log {j+1}_1"
                log2_key = f"Log {j+1}_2"

                if index_key not in data:
                    data[index_key] = []
                if log1_key not in data:
                    data[log1_key] = []
                if log2_key not in data:
                    data[log2_key] = []

                data[index_key].append(log[0])
                data[log1_key].append(log[1])
                data[log2_key].append(log[2])
        else:
            # Ensure all similar log keys are present with None values if similar_logs is None
            for j in range(
                K_VALUE
            ):  # Assuming K_VALUE is the max number of similar logs expected
                index_key = f"Log Pair {j+1} Similarity Index"
                log1_key = f"Log {j+1}_1"
                log2_key = f"Log {j+1}_2"

                if index_key not in data:
                    data[index_key] = []
                if log1_key not in data:
                    data[log1_key] = []
                if log2_key not in data:
                    data[log2_key] = []

                data[index_key].append(None)
                data[log1_key].append(None)
                data[log2_key].append(None)

        # Append dissimilar log pairs for template 1
        if "Similarity Index For Most Dissimilar Log Pair From Template 1" not in data:
            data["Similarity Index For Most Dissimilar Log Pair From Template 1"] = []
        if "Dissimilar Log 1 For Template 1" not in data:
            data["Dissimilar Log 1 For Template 1"] = []
        if "Dissimilar Log 2 For Template 1" not in data:
            data["Dissimilar Log 2 For Template 1"] = []

        data["Similarity Index For Most Dissimilar Log Pair From Template 1"].append(
            dissimilar_log_pair_1[0]
        )
        data["Dissimilar Log 1 For Template 1"].append(dissimilar_log_pair_1[1])
        data["Dissimilar Log 2 For Template 1"].append(dissimilar_log_pair_1[2])

        # Append dissimilar log pairs for template 2
        if "Similarity Index For Most Dissimilar Log Pair From Template 2" not in data:
            data["Similarity Index For Most Dissimilar Log Pair From Template 2"] = []
        if "Dissimilar Log 1 For Template 2" not in data:
            data["Dissimilar Log 1 For Template 2"] = []
        if "Dissimilar Log 2 For Template 2" not in data:
            data["Dissimilar Log 2 For Template 2"] = []

        if dissimilar_log_pair_2 is not None:
            data[
                "Similarity Index For Most Dissimilar Log Pair From Template 2"
            ].append(dissimilar_log_pair_2[0])
            data["Dissimilar Log 1 For Template 2"].append(dissimilar_log_pair_2[1])
            data["Dissimilar Log 2 For Template 2"].append(dissimilar_log_pair_2[2])
        else:
            data[
                "Similarity Index For Most Dissimilar Log Pair From Template 2"
            ].append(None)
            data["Dissimilar Log 1 For Template 2"].append(None)
            data["Dissimilar Log 2 For Template 2"].append(None)

    df = pd.DataFrame(data)

    # Check if the file already exists
    if os.path.exists(output_path):
        os.remove(output_path)  # Remove the existing file

    df.to_excel(output_path, index=False)


def find_most_dissimilar_log_pair(logs: List[str]) -> Tuple[float, str, str]:
    if len(logs) == 1:
        return (-1, logs[0], logs[0])

    # Initialize with a high initial distance
    min_sim_pair = (float("inf"), "", "")

    # Compute the most dissimilar log message pair
    for i in range(len(logs)):
        for j in range(i + 1, len(logs)):
            # edit_distance = computeEditDistance(logs[i], logs[j])
            # normalized_edit_distance = normalizeEditDistance(
            #     logs[i], logs[j], edit_distance
            # )
            normalized_edit_distance = Levenshtein.ratio(logs[i], logs[j])
            if normalized_edit_distance < min_sim_pair[0]:
                min_sim_pair = (normalized_edit_distance, logs[i], logs[j])

    return min_sim_pair


# Inputs:
#   1. n_most_similar_templates: Tuple[sim index, temp1, temp2]
# Output:
#   1. List[Tuple[similarity index, log1, log2]] where log1 belongs to temp1 and log2 belongs to temp2, in descending order by similarity index
def find_K_most_similar_logs_between_similar_log_templates(
    temp1_logs: List[str], temp2_logs: List[str]
) -> List[Tuple[float, str, str]]:
    min_heap = []

    for log1 in temp1_logs:
        for log2 in temp2_logs:
            # edit_distance = computeEditDistance(log1, log2)
            # normalized_edit_distance = normalizeEditDistance(log1, log2, edit_distance)
            normalized_edit_distance = Levenshtein.ratio(log1, log2)
            if len(min_heap) < K_VALUE:
                heapq.heappush(min_heap, (normalized_edit_distance, log1, log2))
            elif normalized_edit_distance > min_heap[0][0]:
                heapq.heapreplace(min_heap, (normalized_edit_distance, log1, log2))

    # Ensure min_heap has exactly K_VALUE entries
    while len(min_heap) < K_VALUE:
        heapq.heappush(min_heap, (-1, "", ""))

    # Convert min_heap to a sorted array in descending order of similarity index
    k_most_similar_logs = sorted(min_heap, key=lambda x: x[0], reverse=True)

    return k_most_similar_logs


# Inputs:
#   1. templates: list of unique log templates
# Output:
#   1. n most similar log templates: List[Tuple[sim index, temp1, temp2]] in descending order of sim index
def find_N_most_similar_templates(
    templates: List[str],
) -> List[Tuple[float, str, str]]:
    min_heap = []  # stores (similarity index, template1, template2)

    # Compute the n most similar log templates
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            # edit_distance = computeEditDistance(templates[i], templates[j])
            # normalized_edit_distance = normalizeEditDistance(
            #     templates[i], templates[j], edit_distance
            # )
            normalized_edit_distance = Levenshtein.ratio(templates[i], templates[j])
            if len(min_heap) < N_VALUE:
                heapq.heappush(
                    min_heap, (normalized_edit_distance, templates[i], templates[j])
                )
            elif normalized_edit_distance > min_heap[0][0]:
                heapq.heapreplace(
                    min_heap, (normalized_edit_distance, templates[i], templates[j])
                )

    while len(min_heap) < N_VALUE:
        heapq.heappush(min_heap, (-1, "", ""))

    # Convert min_heap to a sorted array in descending order of similarity index
    n_most_similar_templates = sorted(min_heap, key=lambda x: x[0], reverse=True)

    return n_most_similar_templates


# Inputs:
#   1. template: log template to find corresponding logs to -> str
#   2. file_name: Name of input file -> str
#   3. log_column_name: The name of the column containing the logs
# Output:
#   1. A list of logs belonging to the log template
def get_log_cluster(template: str, input_path: str) -> List[str]:
    # Read the spreadsheet
    df = pd.read_excel(input_path)

    matching_logs = df[df[TEMPLATE_COLUMN_NAME] == template][LOG_COLUMN_NAME].tolist()

    return matching_logs


# Inputs:
#   1. path: path to excel file to read -> str
#   2. template_column_name: Name of column with log templates -> str
#   3. create_new_file: Whether or not we want to create a new file -> bool
# Output:
#   1. A list of unique [log template]
def remove_duplicate_templates(path: str, template_column_name: str) -> List[str]:
    # Read the spreadsheet
    df = pd.read_excel(path)

    # Keep a copy of the original templates
    df["Original_Template"] = df[template_column_name]

    # Remove duplicate rows based on the "Original_Template" column
    unique_templates_df = df.drop_duplicates(subset=["Original_Template"])

    # Extract the unique original templates into a list
    unique_templates = unique_templates_df["Original_Template"].tolist()

    return unique_templates


# Inputs:
#   1. str1 and str2: two log templates with whitespaces removed
# Output:
#   1. A float from 0 to 1 (inclusive) representing the normalized edit distance between the two templates
#       - closer to 0 means not similar
#       - closer to 1 means similar
# def normalizeEditDistance(str1: str, str2: str, edit_distance: int) -> float:
#     return 1 - (edit_distance) / (max(len(str1), len(str2)))


# Implementation of Levenshtein Distance
# Inputs:
#   1. str1 and str2: two log templates with whitespaces removed
# Output:
#   1. An integer representing the edit distance between the two templates
# def computeEditDistance(str1: str, str2: str) -> int:
#     # create 2D dp array
#     dp = [[float("inf")] * (len(str2) + 1) for i in range(len(str1) + 1)]

#     # base cases
#     for j in range(len(str2) + 1):
#         dp[len(str1)][j] = len(str2) - j
#     for i in range(len(str1) + 1):
#         dp[i][len(str2)] = len(str1) - i

#     # dp step
#     for i in range(len(str1) - 1, -1, -1):
#         for j in range(len(str2) - 1, -1, -1):
#             if str1[i] == str2[j]:
#                 dp[i][j] = dp[i + 1][j + 1]
#             else:
#                 dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
#     return dp[0][0]


if __name__ == "__main__":
    main()
