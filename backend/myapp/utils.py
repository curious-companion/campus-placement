import pandas as pd
import os
import pickle
from uuid import uuid4
import traceback
import logging
# Prepared data in the format required by deployed model


def transform_prediction(df):
    transformed_data = []

    for index, row in df.iterrows():
        data = {
            "cgpa": float(row["cgpa"]),
            "inter_gpa": float(row["inter_gpa"]),
            "ssc_gpa": float(row["ssc_gpa"]),
            "internships": int(row["internships"]),
            "no_of_projects": int(row["no_of_projects"]),
            "is_participate_hackathon": int(row["is_participate_hackathon"]),
            "is_participated_extracurricular": int(row["is_participated_extracurricular"]),
            "no_of_programming_languages": int(row["no_of_programming_languages"]),
            "dsa": int(row["dsa"]),
            "mobile_dev": int(row["mobile_dev"]),
            "web_dev": int(row["web_dev"]),
            "Machine Learning": int(row["Machine Learning"]),
            "cloud": int(row["cloud"]),

            # Tier one-hot
            "tier_1": 1 if row["tier"] == 1 else 0,
            "tier_2": 1 if row["tier"] == 2 else 0,
            "tier_3": 1 if row["tier"] == 3 else 0,

            # Gender one-hot (default to gender_M)
            "gender_F": 0,
            "gender_M": 1,
            "gender_nan": 0,

            # Branch one-hot
            "branch_CSE": int(row.get("CSE", 0)),
            "branch_ECE": int(row.get("ECE", 0)),
            "branch_EEE": int(row.get("EEE", 0)),
            "branch_MECH": int(row.get("MECH", 0)),

        }

        transformed_data.append(data)

    return transformed_data




def load_pickle_models():
    salary_model_path = os.path.join(
        os.path.dirname(__file__), 'models', 'sal_model.pkl')
    placed_model_path = os.path.join(os.path.dirname(
        __file__), 'models', 'Placed_model.pkl')

    salary_model = pickle.load(open(salary_model_path, 'rb'))
    is_placed_model = pickle.load(open(placed_model_path, 'rb'))

    return is_placed_model, salary_model


def save_df_to_temp(df):
    unique_id = str(uuid4())
    temp_file_url_path = os.path.join(
        os.path.dirname(__file__), '..', 'static', 'temp', unique_id+'.csv')

    df.to_csv(temp_file_url_path)

    return '/temp/'+unique_id+'.csv'


def deleteTempFiles():
    directory_path = temp_file_url_path = os.path.join(
        os.path.dirname(__file__), '..', 'static', 'temp')

    # List all files in the directory
    file_list = os.listdir(directory_path)

    # Iterate through the files and delete them
    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(file_path)[1]
                print(file_extension)
                if file_extension != '.txt':
                    os.remove(file_path)

                    print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def convert_is_placed_to_zero_ifnot_placed(is_placed, salary):

    for i in range(len(is_placed)):
        if is_placed[i] == 0:
            salary[i] = 0

    return salary


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')

def check_columns_and_datatypes(excel_file):
    expected_columns = [
        "s_id", "name", "tier", "gender", "branch", "cgpa", "inter_gpa",
        "ssc_gpa", "internships", "no_of_projects", "is_participate_hackathon",
        "is_participated_extracurricular", "no_of_programming_languages",
        "dsa", "mobile_dev", "web_dev", "Machine Learning", "cloud", "other_skills"
    ]
    expected_datatypes = {
        "s_id": int, "name": str, "tier": int, "gender": str, "branch": str,
        "cgpa": float, "inter_gpa": float, "ssc_gpa": float, "internships": int,
        "no_of_projects": int, "is_participate_hackathon": int,
        "is_participated_extracurricular": int, "no_of_programming_languages": int,
        "dsa": int, "mobile_dev": int, "web_dev": int, "Machine Learning": int,
        "cloud": int, "other_skills": str
    }

    try:
        logging.debug("Checking file type...")

        if excel_file.name.endswith(".xlsx"):
            df = pd.read_excel(excel_file)
            logging.debug("Loaded Excel file successfully.")
        elif excel_file.name.endswith(".csv"):
            df = pd.read_csv(excel_file)
            logging.debug("Loaded CSV file successfully.")
        else:
            logging.error("Unsupported file format.")
            return True, 'Please upload .csv or .xlsx file only.'

        logging.debug(f"File contains columns: {list(df.columns)}")

        if set(expected_columns) != set(df.columns):
            logging.error("Column names do not match expected format.")
            return True, 'Column names not matched in the uploaded file.'

        logging.debug("Checking for null values (excluding 'name' and 'other_skills')...")
        null_check = df.drop(columns=['other_skills', 'name']).isnull().values.any()

        if null_check:
            logging.warning("Null values found in required fields.")
            return True, 'File contains empty/null cells. Fill data completely.'

        logging.debug("Checking and converting column data types...")
        for column, datatype in expected_datatypes.items():
            if column in df.columns:
                current_dtype = df[column].dtype
                logging.debug(f"Column '{column}' has dtype {current_dtype}; expected {datatype}.")
                try:
                    if datatype == int:
                        df[column] = pd.to_numeric(df[column], errors='raise').astype(int)
                    elif datatype == float:
                        df[column] = pd.to_numeric(df[column], errors='raise').astype(float)
                    elif datatype == str:
                        df[column] = df[column].astype(str)
                except Exception as e:
                    logging.error(f"Failed to convert column '{column}' to {datatype}. Error: {str(e)}")
                    return True, f'{column} has invalid data type.'

        logging.info("File passed all checks successfully.")
        return False, ''

    except Exception as e:
        logging.critical("An exception occurred during file validation.")
        logging.critical(traceback.format_exc())
        return True, 'Something is wrong in the uploaded file.'

def delete_file(pathname):
    try:
        os.remove(pathname)
        print(f"Deleted file: {pathname}")
    except Exception as e:
        print(f"Error deleting file: {e}")
