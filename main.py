import os
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv
import heapq

# Load environment variables from .env file
load_dotenv()

# File names
INPUT_FILE_NAME = os.getenv("INPUT_FILE_NAME")
OUTPUT_FILE_NAME = os.getenv("OUTPUT_FILE_NAME")
TEMPLATE_COLUMN_NAME = os.getenv("TEMPLATE_COLUMN_NAME")
LOG_COLUMN_NAME = os.getenv("LOG_COLUMN_NAME")
INPUT_PATH = os.getenv("INPUT_FOLDER") + "/" + INPUT_FILE_NAME + ".xlsx"
OUTPUT_PATH = os.getenv("OUTPUT_FOLDER") + "/" + OUTPUT_FILE_NAME + ".xlsx"
N_VALUE = int(os.getenv("N_VALUE"))
K_VALUE = int(os.getenv("K_VALUE"))
VIEW_UNIQUE_TEMPLATES = bool(os.getenv("VIEW_UNIQUE_TEMPLATES"))


def main():
    res = (
        []
    )  # stores a list of (similar template pair, list of similar logs between template pair)

    templates = removeDuplicateTemplates(
        INPUT_PATH, TEMPLATE_COLUMN_NAME, VIEW_UNIQUE_TEMPLATES
    )

    n_most_similar_templates = findNMostSimilarTemplates(templates)

    for pair in n_most_similar_templates:
        # temp = findKMostSimilarLogsBetweenSimilarLogTemplates(pair)
        # print(temp)
        res.append((pair, findKMostSimilarLogsBetweenSimilarLogTemplates(pair)))

    generateOutputFile(res)


def generateOutputFile(
    all_logs: List[Tuple[Tuple[float, str, str], List[Tuple[float, str, str]]]]
) -> None:
    # print(all_logs[0])
    data = {
        "Template Similarity Index": [],
        "Template 1": [],
        "Template 2": [],
    }

    # Dynamically add log pairs
    for pair, logs in all_logs:
        data["Template Similarity Index"].append(pair[0])
        data["Template 1"].append(pair[1])
        data["Template 2"].append(pair[2])

        for j, log in enumerate(logs):
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

    df = pd.DataFrame(data)

    # Check if the file already exists
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)  # Remove the existing file

    df.to_excel(OUTPUT_PATH, index=False)


# Inputs:
#   1. n_most_similar_templates: Tuple[sim index, temp1, temp2]
# Output:
#   1. List[Tuple[similarity index, log1, log2]] where log1 belongs to temp1 and log2 belongs to temp2, in descending order by similarity index
def findKMostSimilarLogsBetweenSimilarLogTemplates(
    pair: Tuple[float, str, str]
) -> List[Tuple[float, str, str]]:
    # Clear the max_heap
    max_heap = []

    temp1 = pair[1]
    temp2 = pair[2]

    temp1_logs = getLogCluster(temp1)
    temp2_logs = getLogCluster(temp2)

    for log1 in temp1_logs:
        for log2 in temp2_logs:
            edit_distance = computeEditDistance(log1, log2)
            normalized_edit_distance = normalizeEditDistance(log1, log2, edit_distance)
            heapq.heappush(
                max_heap,
                (-normalized_edit_distance, log1, log2),
            )

    k_most_similar_logs = getNMostSimilar(K_VALUE, max_heap)

    return k_most_similar_logs


# Inputs:
#   1. templates: list of unique log templates
# Output:
#   1. n most similar log templates: List[Tuple[sim index, temp1, temp2]] in descending order of sim index
def findNMostSimilarTemplates(templates: List[str]) -> List[Tuple[float, str, str]]:
    max_heap = []  # stores (-similarity index, template1, template2)

    # Compute the n most similar log templates
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            edit_distance = computeEditDistance(templates[i], templates[j])
            normalized_edit_distance = normalizeEditDistance(
                templates[i], templates[j], edit_distance
            )
            heapq.heappush(
                max_heap, (-normalized_edit_distance, templates[i], templates[j])
            )
    n_most_similar_templates = getNMostSimilar(N_VALUE, max_heap)
    # createFileForNMostSimilarTemplates(n_most_similar_templates, OUTPUT_PATH)

    return n_most_similar_templates


# Inputs:
#   1. template: log template to find corresponding logs to -> str
#   2. file_name: Name of input file -> str
#   3. log_column_name: The name of the column containing the logs
# Output:
#   1. A list of logs belonging to the log template
def getLogCluster(template: str) -> List[str]:
    # Read the spreadsheet
    df = pd.read_excel(INPUT_PATH)

    matching_logs = df[df[TEMPLATE_COLUMN_NAME] == template][LOG_COLUMN_NAME].tolist()

    return matching_logs


# Inputs:
#   1. path: path to excel file to read -> str
#   2. template_column_name: Name of column with log templates -> str
#   3. create_new_file: Whether or not we want to create a new file -> bool
# Output:
#   1. A list of unique [log template]
def removeDuplicateTemplates(
    path: str, template_column_name: str, create_new_file: bool
) -> List[str]:
    # Read the spreadsheet
    df = pd.read_excel(path)

    # Keep a copy of the original templates
    df["Original_Template"] = df[template_column_name]

    # Remove duplicate rows based on the "Original_Template" column
    unique_templates_df = df.drop_duplicates(subset=["Original_Template"])

    # Extract the unique original templates into a list
    unique_templates = unique_templates_df["Original_Template"].tolist()

    if create_new_file:
        # Save the DataFrame with the original templates to a new Excel file
        unique_templates_df.to_excel(
            path.replace(".xlsx", "_unique_templates.xlsx"), index=False
        )

    return unique_templates


# Inputs:
#   1. templates: An array of the n most similar templates sorted by ascending similarity index
#   2. file_name: output file name
# Output:
#   1. A .xlsx file containing the n templates ranked from most similar (top) to least similar (bottom)
# def createFileForNMostSimilarTemplates(
#     templates: List[Tuple[float, str, str]], file_name: str
# ) -> None:
#     # Create a DataFrame with the required columns
#     df = pd.DataFrame(
#         templates, columns=["Similarity Index", "Template 1", "Template 2"]
#     )

#     # Check if the file already exists
#     if os.path.exists(file_name):
#         os.remove(file_name)  # Remove the existing file

#     # Save the DataFrame to an Excel file
#     df.to_excel(file_name, index=False)


# Inputs:
#   1. n: An integer representing the number of most similar templates to return
#   2. max_heap: The max heap containing pairs of log templates and their similarity index
# Output:
#   1. An array of the n most similar templates based on similarity index
def getNMostSimilar(
    n: int, max_heap: List[Tuple[float, str, str]]
) -> List[Tuple[float, str, str]]:
    res = []
    temp_heap = max_heap[:]
    heapq.heapify(temp_heap)

    for _ in range(min(n, len(temp_heap))):
        similarity_index, template1, template2 = heapq.heappop(temp_heap)
        res.append((-similarity_index, template1, template2))

    return res


# Inputs:
#   1. str1 and str2: two log templates with whitespaces removed
# Output:
#   1. A float from 0 to 1 (inclusive) representing the normalized edit distance between the two templates
#       - closer to 0 means not similar
#       - closer to 1 means similar
def normalizeEditDistance(str1: str, str2: str, edit_distance: int) -> float:
    return 1 - (edit_distance) / (max(len(str1), len(str2)))


# Implementation of Levenshtein Distance
# Inputs:
#   1. str1 and str2: two log templates with whitespaces removed
# Output:
#   1. An integer representing the edit distance between the two templates
def computeEditDistance(str1: str, str2: str) -> int:
    # create 2D dp array
    dp = [[float("inf")] * (len(str2) + 1) for i in range(len(str1) + 1)]

    # base cases
    for j in range(len(str2) + 1):
        dp[len(str1)][j] = len(str2) - j
    for i in range(len(str1) + 1):
        dp[i][len(str2)] = len(str1) - i

    # dp step
    for i in range(len(str1) - 1, -1, -1):
        for j in range(len(str2) - 1, -1, -1):
            if str1[i] == str2[j]:
                dp[i][j] = dp[i + 1][j + 1]
            else:
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
    return dp[0][0]


if __name__ == "__main__":
    main()
