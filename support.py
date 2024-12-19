import pickle
import pandas as pd

try:
    # Load the CSV file
    df = pd.read_csv('data/BX-Books.csv', encoding='latin-1', on_bad_lines='skip')

    # Print columns to verify
    print("Columns in the DataFrame:")
    print(df.columns)

    # Normalize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    print("Normalized columns:")
    print(df.columns)

    # Access a sample row
    sample_row = df.iloc[0]
    print("Sample row data:")
    print(sample_row)

    # Example of accessing the 'book-title' column
    print("Book Title:", sample_row['book_title'])

except KeyError as e:
    print(f"Column not found: {e}")
except Exception as e:
    print(f"Error: {e}")

books_image={}
for index, row in df.iterrows():
    books_image[row['ISBN']]={
        "name":row['Book-Title'],
        "image":row['Image-URL-M']
    }

with open("model/books_image.pkl", "wb") as f:
    pickle.dump(books_image,f)