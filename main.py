import os
from typing import List
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File names
FILE_NAME = os.getenv("EXCEL_FILE_NAME")
TEMPLATE_COLUMN_NAME = os.getenv("TEMPLATE_COLUMN_NAME")
PATH = FILE_NAME + ".xlsx"


def main():
    templates = removeDuplicateTemplates(PATH, TEMPLATE_COLUMN_NAME, False)


# Inputs:
#   1. path: path to excel file to read -> str
#   2. template_column_name: Name of column with log templates -> str
#   3. create_new_file: Whether or not we want to create a new fil -> bool
# Output:
#   1. A list of unique log templates with whitespaces removed
def removeDuplicateTemplates(
    path: str, template_column_name: str, create_new_file: bool
) -> List[str]:
    # Read the spreadsheet
    df = pd.read_excel(path)

    # Remove whitespaces in each row of the "Template" column
    df[template_column_name] = df[template_column_name].str.replace(" ", "")

    # Remove duplicate rows based on the "Templates" column
    unique_templates = df[template_column_name].drop_duplicates().tolist()

    if create_new_file:
        # Create a new DataFrame with the unique templates
        unique_df = pd.DataFrame(unique_templates[template_column_name])

        # Rename the column
        unique_df.columns = ["Unique_Templates"]

        # Save the new DataFrame to a new Excel file
        unique_df.to_excel(FILE_NAME + "_unique_templates.xlsx", index=False)

    return unique_templates


if __name__ == "__main__":
    main()
