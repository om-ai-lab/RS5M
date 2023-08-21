from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import csv

app = Flask(__name__)

# Load the data
# image_text_pairs = [("static/00001.jpg", "A satellite image of mountains and village next to them."),
#                     ("static/00001.jpg", "Image 2 description"),
#                     # Add as many pairs as you need
#                     ]

image_text_pairs = []
with open("rs3_dump.csv", "r", encoding='utf8', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == "name":
            continue
        name = "static/" + row[0]
        text = row[1]
        image_text_pairs.append((name, text))
print(len(image_text_pairs))

# Initialize the current index
index = [0]

# Initialize the DataFrame to save the ratings
df = pd.DataFrame(columns=["image_path", "description", "relevance_detail", "hallucination", "fluency"])


@app.route('/', methods=['GET', 'POST'])
def rate():
    if request.method == 'POST':
        # Record the ratings
        ratings = {
            "image_path": image_text_pairs[index[0]][0],
            "description": image_text_pairs[index[0]][1],
            "relevance_detail": request.form.get('relevance_detail'),
            "hallucination": request.form.get('hallucination'),
            "fluency": request.form.get('fluency'),
        }
        print(ratings)
        global df
        df = df.append(ratings, ignore_index=True)

        # Save the DataFrame to a csv file
        df.to_csv('ratings.csv', index=False)

        # Move to the next pair
        index[0] += 1
        if index[0] >= len(image_text_pairs):
            print("Done")
            exit()

    image_path, description = image_text_pairs[index[0]]
    return render_template('index.html', image_path=image_path, description=description)


if __name__ == "__main__":
    app.run(debug=True)
