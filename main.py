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
INPUT_PATH = os.getenv("INPUT_FOLDER") + "/" + INPUT_FILE_NAME + ".xlsx"
OUTPUT_PATH = os.getenv("OUTPUT_FOLDER") + "/" + OUTPUT_FILE_NAME + ".xlsx"
N_VALUE = int(os.getenv("N_VALUE"))
VIEW_UNIQUE_TEMPLATES = bool(os.getenv("VIEW_UNIQUE_TEMPLATES"))


def main():
    min_heap = []  # stores (-similarity index, template1, template2)
    templates = removeDuplicateTemplates(
        INPUT_PATH, TEMPLATE_COLUMN_NAME, VIEW_UNIQUE_TEMPLATES
    )
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            edit_distance = computeEditDistance(templates[i][1], templates[j][1])
            heapq.heappush(min_heap, (edit_distance, templates[i][0], templates[j][0]))
    n_most_similar_templates = getNMostSimilarTemplates(N_VALUE, min_heap)
    createFileForNMostSimilarTemplates(n_most_similar_templates, OUTPUT_PATH)


# Inputs:
#   1. path: path to excel file to read -> str
#   2. template_column_name: Name of column with log templates -> str
#   3. create_new_file: Whether or not we want to create a new fil -> bool
# Output:
#   1. A list of unique [original log template, log template with whitespaces removed]
def removeDuplicateTemplates(
    path: str, template_column_name: str, create_new_file: bool
) -> List[Tuple[str, str]]:
    # Read the spreadsheet
    df = pd.read_excel(path)

    # Keep a copy of the original templates
    df["Original_Template"] = df[template_column_name]

    # Remove whitespaces in each row of the "Template" column
    df["Modified_Template"] = df[template_column_name].str.replace(" ", "")

    # Remove duplicate rows based on the "Modified_Template" column
    unique_templates_df = df.drop_duplicates(subset=["Modified_Template"])

    unique_templates = list(
        unique_templates_df[["Original_Template", "Modified_Template"]].itertuples(
            index=False, name=None
        )
    )

    if create_new_file:
        # Save the DataFrame with the original and modified templates to a new Excel file
        unique_templates_df.to_excel(
            INPUT_FILE_NAME + "_unique_templates.xlsx", index=False
        )

    return unique_templates


def createFileForNMostSimilarTemplates(
    templates: List[Tuple[float, str, str]], file_name: str
) -> None:
    # Create a DataFrame with the required columns
    df = pd.DataFrame(
        templates, columns=["Similarity Index", "Template 1", "Template 2"]
    )

    # Check if the file already exists
    if os.path.exists(file_name):
        os.remove(file_name)  # Remove the existing file

    # Save the DataFrame to an Excel file
    df.to_excel(file_name, index=False)


# Inputs:
#   1. n: An integer representing the number of most similar templates to return
#   2. min_heap: The max heap containing pairs of log templates and their similarity index
# Output:
#   1. An array of the n most similar templates based on similarity index
def getNMostSimilarTemplates(
    n: int, min_heap: List[Tuple[float, str, str]]
) -> List[Tuple[float, str, str]]:
    res = []
    temp_heap = min_heap[:]
    heapq.heapify(temp_heap)

    for _ in range(min(n, len(temp_heap))):
        similarity_index, template1, template2 = heapq.heappop(temp_heap)
        res.append((similarity_index, template1, template2))

    return res


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
