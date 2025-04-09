import json
import pandas as pd


def main():
    # step 1: open and read the jsonl file
    all_data = []
    with open("data/train.jsonl") as input_file:
        for each_line in input_file:
            all_data.append(json.loads(each_line))
    
    # step 2: extract and store useful fields 
    file_name = []
    function_name = []
    version_data = []
    diff_code = []
    diff_docstring = []
    
    for each_row in all_data:
        file_name.append(each_row['file'])
        function_name.append(each_row['function'])
        
        latest_version = 0
        latest_version_index = 0
        for i, each_version in enumerate(each_row["version_data"]):
            if int(list(each_version.keys())[0].split("v")[1]) > latest_version:
                latest_version = int(list(each_version.keys())[0].split("v")[1])
                latest_version_index = i
                
        version_data.append(each_row["version_data"][latest_version_index][f"v{latest_version}"])
        diff_code.append(each_row["diff_code"])
        diff_docstring.append(each_row["diff_docstring"])
    
       
    # store useful fields in dataframe
    dataframe_dict = {
        "file": file_name,
        "function": function_name,
        "version": version_data,
        "diff_code": diff_code,
        "diff_docstring": diff_docstring
    } 
    
    df = pd.DataFrame(dataframe_dict)
    
    # converting to string to check the duplicate value
    df["version"] = df["version"].apply(str)
    count_duplicates = df.duplicated().sum()
    print(count_duplicates)  # It verifies that there is no duplicate value in datasets
    
    # count the number of null values
    count_null = df.isnull().sum()
    print(count_null)  # It verifies the datasets is pretty clean as there is no null value as well
    
    

if __name__ == "__main__":
    main()
    