# Video-Game-Search-Engine
Information Retrieval Summative Coursework 
Instructions to run

1. Make sure to have the correct libararies installed
    Pip install:
    - math
    - os
    - pickle
    - pandas
    - nltk 
    - bs4
    - numpy
    - matplotlib
    - matplotlib_venn

2. Run the create_TFIDF.py file and wait until the matrix is pickled - should display a success message at bottom

3. Run the query_TFIDF.py file and follow the instructions on the terminal
    You can:
    - Enter your query to search on the videogame files
    - Select a way of calculating the weighting for the queried files (A, B, C)
        - For "option C" you can enhance the query using relevance feedback. Each document will have its ID at the beginning. You can type the ID you want as the document you, as the user thinks is relevant.
    - The process will then begin again with you choosing your query
    - To end the process - just press Enter on the query search input.

4. The batch_query_TFIDF works in a similar way to the regular query_TFIDF, but has automated queries rather than a user input.
   This is to allow fast testing and generating graphs for the presentation. It runs automatically.