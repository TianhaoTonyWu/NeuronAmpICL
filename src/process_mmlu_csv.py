import os
import pandas as pd

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "stem": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "others": ["other", "business", "health"],
}

old_test_dir = f"/home/wth/few_vs_zero/datasets/MMLU/test"
old_dev_dir =  f"/home/wth/few_vs_zero/datasets/MMLU/dev"


train_output_dir = f"/home/wth/few_vs_zero/datasets/MMLU/merged_train"
dev_output_dir = f"/home/wth/few_vs_zero/datasets/MMLU/merged_dev"
test_output_dir = f"/home/wth/few_vs_zero/datasets/MMLU/new_test"
val_output_dir = f"/home/wth/few_vs_zero/datasets/MMLU/new_val"

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)
os.makedirs(dev_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Create a dictionary to store dataframes for each category
merged_train_data = {cat: [] for cat in categories.keys()}
merged_dev_data = {cat: [] for cat in categories.keys()}  
test_data = {cat: [] for cat in subcategories.keys()} 
val_data = {cat: [] for cat in subcategories.keys()} 


# Loop through each subcategory
for subcategory, subjects in subcategories.items():
    filename = os.path.join(old_test_dir, f"{subcategory}_test.csv")
    
    # Read the CSV file
    if os.path.exists(filename):
        df = pd.read_csv(filename, delimiter=",", header=None)
        print(f"Loaded {filename}: {df.shape[0]} rows, Columns: {df.columns.tolist()}")
        
        midpoint = len(df) // 2

        # Split the DataFrame
        train_df = df.iloc[:midpoint]  # First half
        test_df = df.iloc[midpoint:]  # Second half

        # Determine the category based on the subjects
        for category, category_subjects in categories.items():
            if any(subject in subjects for subject in category_subjects):
                merged_train_data[category].append(train_df)
        
        # Save the other half as test
        test_data[subcategory].append(test_df)

        # keep this half as validation for each subcategory
        val_data[subcategory].append(train_df)
              

# Merge Dev
for subcategory, subjects in subcategories.items():
    filename = os.path.join(old_dev_dir, f"{subcategory}_dev.csv")
    
    # Read the CSV file
    if os.path.exists(filename):
        dev_df = pd.read_csv(filename, delimiter=",", header=None)
        print(f"Loaded {filename}: {df.shape[0]} rows, Columns: {df.columns.tolist()}")
        
        # Determine the category based on the subjects
        for category, category_subjects in categories.items():
            if any(subject in subjects for subject in category_subjects):
                merged_dev_data[category].append(dev_df)





for subcategory, dfs in test_data.items():
    if dfs and len(dfs) == 1:
        output_test_file = os.path.join(test_output_dir, f"{subcategory}_test.csv")
        test_df = dfs[0]
        test_df.to_csv(output_test_file, index=False)
        print(f"Saved test file for category '{subcategory}' with {test_df.shape[0]} rows, Columns: {test_df.columns.tolist()}")

for subcategory, dfs in val_data.items():
    if dfs and len(dfs) == 1:
        output_val_file = os.path.join(val_output_dir, f"{subcategory}_val.csv")
        val_df = dfs[0]
        val_df.to_csv(output_val_file, index=False)
        print(f"Saved val file for category '{subcategory}' with {val_df.shape[0]} rows, Columns: {val_df.columns.tolist()}")

# Merge the dataframes for each category and save to new CSV files
for category, dataframes in merged_train_data.items():
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        output_train_file = os.path.join(train_output_dir, f"{category}_train.csv")
        merged_df.to_csv(output_train_file, index=False)
        print(f"Saved merged training file for category '{category}' with {merged_df.shape[0]} rows, Columns: {merged_df.columns.tolist()}")

for category, dataframes in merged_dev_data.items():
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        output_test_file = os.path.join(dev_output_dir, f"{category}_dev.csv")
        merged_df.to_csv(output_test_file, index=False)
        print(f"Saved merged dev file for category '{category}' with {merged_df.shape[0]} rows, Columns: {merged_df.columns.tolist()}")
