import pandas as pd
from groq import Groq

filepath = "resumes_with_keywords_main.csv"
def read_csv(file_path):
    df = pd.read_csv(file_path)  # Read CSV into DataFrame
    return df

def generate_questions_for_rows(file_path, output_file):
    df = read_csv(file_path)

    client = Groq(api_key="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")

    with open(output_file, "w", encoding="utf-8") as file:
        for index, row in df.iterrows():
            keywords = row['Extracted Keywords']

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the keywords and generate 2 question for only two keywords not more than two."
                    },
                    {
                        "role": "user",
                        "content": keywords  # Pass the keyword of the current row
                    }
                ],
                model="llama3-8b-8192",

            )

            questions = chat_completion.choices[0].message.content
            file.write(f"Questions for Row {index + 1} based on keywords:\n{questions}\n\n")
            print(f"Questions for Row {index + 1} saved.")
def main():
    
    output_file = "generated_questions_main.txt"
    generate_questions_for_rows(filepath, output_file)  
    print(f"All questions have been saved in {output_file}.")
if __name__ == "__main__":
    main()
