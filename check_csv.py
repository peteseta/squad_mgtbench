import csv

def check_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        line_number = 0
        error_lines = []

        for row in reader:
            line_number += 1
            if len(row) > 10:
                error_lines.append(line_number)
                
        if error_lines:
            print(f"Lines with more than 10 columns: {error_lines}")
        else:
            print("No lines found with more than 10 columns.")

if __name__ == "__main__":
    check_csv("SQuAD2_LLMs.csv")
