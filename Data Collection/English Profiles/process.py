import pandas as pd
from fuzzywuzzy import fuzz, process
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from datetime import datetime
import csv

""" # Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True) """

def main():
    # Step 1: Combine CSV files
    print("Step 1: Combining CSV files...")
    
    # List of your CSV files
    files = ["NEW.csv"]

    # Columns you want to keep
    columns_to_keep = ["firstName", "lastName", "headline", "currentPosition/0/companyName",
                       "location/parsed/country","about","connectionsCount",
                       "experience/0/position","experience/0/startDate/text","experience/0/endDate/text",
                       "experience/1/position","experience/1/startDate/text","experience/1/endDate/text",
                       "experience/2/position","experience/2/startDate/text","experience/2/endDate/text",
                       "experience/3/position","experience/3/startDate/text","experience/3/endDate/text",
                       "experience/4/position","experience/4/startDate/text","experience/4/endDate/text",
                       "experience/5/position","experience/5/startDate/text","experience/5/endDate/text",
                       "experience/6/position","experience/6/startDate/text","experience/6/endDate/text",
                       "experience/7/position","experience/7/startDate/text","experience/7/endDate/text",
                       "experience/8/position","experience/8/startDate/text","experience/8/endDate/text",
                       "experience/9/position","experience/9/startDate/text","experience/9/endDate/text",
                       "experience/10/position","experience/10/startDate/text","experience/10/endDate/text",
                       "experience/11/position","experience/12/startDate/text","experience/11/endDate/text",
                       "skills/0/name","skills/1/name","skills/2/name","skills/3/name","skills/4/name","skills/5/name",
                       "languages/0/name","languages/1/name","languages/2/name","languages/3/name",
                       "premium","verified","linkedinUrl"]

    # Read, filter, and combine all CSVs
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            
            # Add missing columns with empty values
            for col in columns_to_keep:
                if col not in df.columns:
                    df[col] = ""
            
            # Keep only the desired columns in the correct order
            df = df[columns_to_keep]
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File {f} not found. Skipping...")
            continue

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates based on linkedinUrl
    combined_df = combined_df.drop_duplicates(subset=["linkedinUrl"])

    print("Step 1 completed: CSVs combined, missing columns created, and duplicates removed!")
    
    # Step 2: Add target titles based on fuzzy matching
    print("Step 2: Adding target titles...")
    
    # Example: titles you searched for
    searched_titles = ["Maintenance Engineer",
        "Reliability Engineer",
        "Industrial Maintenance Engineer",
        "Preventive Maintenance Engineer",
        "Predictive Maintenance Engineer",
        "Maintenance & Reliability Engineer",
        "Maintenance Project Engineer",
        "Maintenance Improvement Engineer",
        "Maintenance Planning Engineer",
        "Maintenance Performance Engineer",
        "Maintenance & Inspection Engineer",
        "Maintenance Systems Engineer",
        "Maintenance Support Engineer",
        "Maintenance Process Engineer",
        "Maintenance and Operations Engineer",
        "Equipment Reliability Engineer",
        "Condition Monitoring Engineer",
        "Continuous Improvement Maintenance Engineer",
        "Maintenance Optimization Engineer",
        "Maintenance Coordinator",
        "Reliability Coordinator",
        "Maintenance Methods Coordinator",]

    # Function to find the best matching title
    def match_title(headline):
       if pd.isna(headline):
        return ""  # or "" if you prefer
       headline_str = str(headline)
       match, score = process.extractOne(headline_str, searched_titles)
       return match
    # Collect current positions (example: experience/0/position is usually current)
    combined_df['Target Title'] = combined_df['headline'].apply(match_title)

    print("Step 2 completed: Added searchedTitle based on current/previous positions!")
    
    # Step 3: Process experiences and clean data
    print("Step 3: Processing experiences and cleaning data...")
    
    def extract_current_positions(row, max_experiences=12):
        """
        Extracts all current positions (where endDate = 'present').
        """
        current_positions = []

        for i in range(max_experiences):
            end_col = f"experience/{i}/endDate/text"
            pos_col = f"experience/{i}/position"

            if end_col not in row:
                continue

            end = row[end_col]
            position = row[pos_col] if pos_col in row else None

            if isinstance(end, str) and end.lower() == "present" and position:
                current_positions.append(position)

        return " | ".join(current_positions) if current_positions else None

    def parse_date_from_text(date_text, default_day=1, default_month=1):
        """
        Parse a date string like 'Dec 2023' or '2023' into a datetime object.
        If month is missing, assumes default_month (January = 1).
        """
        if pd.isna(date_text):
            return None
        date_text = str(date_text).strip()
        if date_text.lower() == "present":
            return datetime.today()
        parts = date_text.split()
        try:
            if len(parts) == 2:  # e.g., "Dec 2023"
                month = datetime.strptime(parts[0][:3], "%b").month
                year = int(parts[1])
            elif len(parts) == 1 and parts[0].isdigit():  # e.g., "2023"
                month = default_month  # January
                year = int(parts[0])
            else:
                return None
            return datetime(year, month, default_day)
        except:
            return None

    def calculate_years_of_experience_from_text(row, max_experiences=12, today=None):
        """
        Calculates total years of experience by parsing start/end from text,
        merging overlapping jobs, and counting months accurately.
        """
        if today is None:
            today = datetime.today()
            
        intervals = []

        for i in range(max_experiences):
            start_col = f"experience/{i}/startDate/text"
            end_col   = f"experience/{i}/endDate/text"

            if start_col not in row or end_col not in row:
                continue

            start_date = parse_date_from_text(row[start_col], default_month=1)
            end_date = parse_date_from_text(row[end_col], default_month=1)

            if start_date is None or end_date is None:
                continue

            if start_date <= end_date:
                intervals.append((start_date, end_date))

        if not intervals:
            return 0.0

        # Sort and merge overlapping intervals
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (min(last_start, current_start), max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        # Calculate total months including both start and end months
        total_months = 0
        for start, end in merged:
            months = (end.year - start.year) * 12 + (end.month - start.month) + 1
            total_months += months

        total_years = round(total_months / 12, 2)
        return total_years

    def process_experiences_years(row):
        """
        Wrapper function that returns both current positions
        and number of years of experience.
        """
        return pd.Series({
            "Current position": extract_current_positions(row),
            "Number of years of experience": calculate_years_of_experience_from_text(row)
        })

    # Apply function to all rows
    combined_df[["Current position", "Number of years of experience"]] = combined_df.apply(process_experiences_years, axis=1)
    assumed_start_age = 24
    combined_df["Estimated age"] = combined_df["Number of years of experience"] + assumed_start_age
    
    # Prepare stopwords + lemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def extract_keywords(text):
        if not isinstance(text, str):
            return "  "

        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove non-alphabetic
        tokens = [re.sub(r"[^a-z]", "", t) for t in tokens if t.isalpha()]
        # Remove stopwords
        tokens = [t for t in tokens if t not in stop_words]

        # POS tagging
        tagged = pos_tag(tokens)

        # Keep only nouns (NN, NNS, NNP, NNPS)
        nouns = [word for word, pos in tagged if pos.startswith("NN")]

        # Lemmatize
        nouns = [lemmatizer.lemmatize(w) for w in nouns]

        return " | ".join(nouns)
    
    combined_df["Keywords"] = combined_df["about"].apply(lambda x: extract_keywords(x).split(" | ") if isinstance(x, str) else [])

    def filter_by_age(df, age_column="Estimated age", max_age=45):
        """
        Returns a DataFrame excluding profiles with age > max_age
        """
        return df[df[age_column] <= max_age].copy()

    # Function 2: Remove profiles with headline containing 'retired' or 'aircraft'
    def filter_by_headline(df, headline_column="headline", keywords=["retired", "aircraft"]):
        """
        Returns a DataFrame excluding profiles whose headline contains any of the keywords (case-insensitive)
        """
        pattern = "|".join(keywords)
        mask = ~df[headline_column].str.contains(pattern, case=False, na=False)
        return df[mask].copy()
    
    # Headline condition: must contain "maintenance" or "reliability"
    def filter_by_headline2(dataframe):
        return dataframe[dataframe["headline"].str.contains("maintenance|reliability", case=False, na=False)]

    combined_df = filter_by_headline2(combined_df)
    combined_df = filter_by_age(combined_df)
    combined_df = filter_by_headline(combined_df)

    print("Step 3 completed: Data processed and cleaned!")
    
    # Step 4: Select final columns and save
    print("Step 4: Selecting final columns and saving...")
    
    # Keep only the columns you want
    final_df = combined_df[["firstName", "lastName", "headline","Target Title","Current position", "currentPosition/0/companyName",
             "location/parsed/country","about","Estimated age","Number of years of experience","connectionsCount",
             "skills/0/name","skills/1/name","skills/2/name","skills/3/name","skills/4/name","skills/5/name",
          "languages/0/name","languages/1/name","languages/2/name","languages/3/name",
            "premium","verified",
            "Keywords", "linkedinUrl"]]

    # Save the final result
    final_df.to_csv("final_cleaned_dataset", index=False, quoting=csv.QUOTE_ALL)
    
    print("All steps completed! Final file saved as 'final_cleaned_dataset.csv'")
    print(f"Final dataset shape: {final_df.shape}")

if __name__ == "__main__":
    main()