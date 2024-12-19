from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pickle files
book_pivot = pickle.load(open("model/book_pivot.pkl", "rb"))
books_name = pickle.load(open("model/books_name.pkl", "rb"))
final_ratings = pickle.load(open("model/final_ratings.pkl", "rb"))
model = pickle.load(open("model/model.pkl", "rb"))

# Recommendation function
def recommend_books(book_name):
    book_name = book_name.strip().lower()
    book_pivot.index = book_pivot.index.str.strip().str.lower()

    # Find the book index
    book_ids = np.where(book_pivot.index == book_name)[0]
    if len(book_ids) == 0:
        return ["Book not found. Please try another book."]
    
    book_id = book_ids[0]
    _, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    recommended_books = []
    for i in suggestions[0]:
        recommended_books.append(books_name[i])
    return recommended_books[1:]  # Exclude the input book itself

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["book_name"]
        recommendations = recommend_books(user_input)
        return render_template("result.html", books=recommendations, input_book=user_input)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
