### 1\. Database Setup and Data Insertion

The script demonstrates how to set up a local database using Python's built-in `sqlite3` library.

  * **Libraries**: The script begins by importing `sqlite3` for interacting with the database and `datetime` and `timedelta` for managing dates.
  * **Connection**: It establishes a connection to a database file named `office_data.db`. If the file does not exist, SQLite creates it automatically. A cursor object is created to execute SQL commands.
    ```python
    import sqlite3
    from datetime import datetime, timedelta

    conn = sqlite3.connect('office_data.db')
    cursor = conn.cursor()
    ```
  * **Table Creation**: An `attendance` table is created to store attendance records.
      * `id`: An integer that serves as the `PRIMARY KEY` and `AUTOINCREMENT`s for each new record, ensuring a unique identifier for every entry.
      * `name`: A text field for the person's name.
      * `date`: A text field for the date of attendance.
    <!-- end list -->
    ```sql
    CREATE TABLE attendance(id integer primary key autoincrement, name text, date text)
    ```
  * **Sample Data Insertion**: The script generates and inserts sample attendance data for three consecutive days. The number of records for each day is dynamic (`100 + i * 10`). `datetime.now()` provides the current date, and `timedelta` is used to calculate past dates.
    ```python
    today = datetime.now()
    for i in range(3):
      day = today - timedelta(days = i)
      for j in range(100 + i * 10):
        cursor.execute("insert into attendance (name, date) values (?, ?)", (f"person_{j+1}" , day.strftime("%Y-%m-%d")))
    conn.commit()
    conn.close()
    ```
  * **Data Verification**: After insertion, the script reconnects to the database, executes a `SELECT` query to fetch the first 5 records, and prints them to verify that the data has been stored correctly.
    ```python
    conn = sqlite3.connect('office_data.db')
    cursor = conn.cursor()
    cursor.execute("select * from attendance limit 5")
    rows = cursor.fetchall()
    for row in rows:
      print(row)
    conn.commit()
    conn.close()
    ```
    An example output would be: `(1, 'person_1', '2025-08-19')`.

-----

### 2\. Querying the Database

The script provides a function to count attendance records for a given day.

  * **`get_attendance_count` Function**: This function takes a `date_str` as input, connects to the database, and executes a `SELECT COUNT(*)` query to count all records that match the given date. It uses a parameterized query (`?`) to prevent SQL injection vulnerabilities. `cursor.fetchone()` retrieves a single row from the query result, and `[0]` is used to access the integer count from the resulting tuple.
    ```python
    def get_attendance_count(date_str):
        conn = sqlite3.connect('office_data.db')
        cursor = conn.cursor()
        cursor.execute("select count(*) from attendance where date = ?", (date_str,))
        count = cursor.fetchone()
        conn.commit()
        conn.close()
        return count
    ```
  * **Example Usage**: Calling the function with a date string returns the attendance count for that day as a tuple, e.g., `(110,)`.

-----

### 3\. Rule-Based Chatbot Implementation

This section demonstrates building a simple chatbot interface that interprets user queries and responds with data from the database.

  * **Simple Chatbot (`handle_query`)**: This initial version of the chatbot is limited to answering only about attendance for "yesterday". It uses an `if 'yesterday' in user_query.lower()` condition to determine the query's intent.
    ```python
    def handle_query(user_query):
      if 'yesterday' in user_query.lower():
        date = (datetime.now() - timedelta(days = 1)).strftime("%Y-%m-%d")
        count = get_attendance_count(date)
        return f"Bot: Yesterday, {count} people were in the office."
      else:
        return "Bot: I can only answer about yesterday count as of now."
    ```
  * **Advanced Chatbot (`chatbot_query_handler`)**: This is a more sophisticated version that can handle various date-related queries using a dedicated date extraction function.
      * **Date Extraction Function**: The `extract_date_from_query` function uses the `re` module (regular expressions) to search for date patterns (`\d{4}-\d{2}-\d{2}`) within the user's query. It also recognizes keywords like "today" and "yesterday".
        ```python
        import re
        def extract_date_from_query(user_query):
          today = datetime.now().date()
          if "today" in user_query.lower():
            return today.strftime("%Y-%m-%d")
          elif "yesterday" in user_query.lower():
            yesterday = today - timedelta(days = 1)
            return yesterday.strftime("%Y-%m-%d")
          else:
            date_match = re.search(r"\d{4}-\d{2}-\d{2}" , user_query)
            if date_match:
              return date_match.group()
            return None
        ```
      * **Query Handler Logic**: The `chatbot_query_handler` function uses `extract_date_from_query` to get the target date. It then retrieves the attendance count for that date and returns a formatted response. It also handles cases where no date is found or no attendance records exist for the specified date.
      * **Demonstration**: A list of sample queries is used to demonstrate the enhanced chatbot's ability to handle different inputs and provide appropriate responses.
        ```python
        querries = [
            "How many people were in the office today?",
            "How many people were in the office on 2025-08-17?",
            #... and other examples
        ]
        for query in querries:
          print(f"User: {query}")
          response = chatbot_query_handler(query)
          print(response)
        ```
      * **Interactive Usage**: The final part of the script uses `input()` to allow for a live, interactive demonstration of the chatbot's functionality.

-----

### 4\. Summary of Key Concepts

  * **SQLite**: A lightweight, serverless database engine that stores data in a single file, making it ideal for local and small-scale applications.
  * **SQL Queries**: The script demonstrates fundamental SQL commands for database management: `CREATE TABLE` to define a table, `INSERT` to add data, and `SELECT COUNT(*)` to retrieve a specific aggregate value.
  * **Parameterized Queries**: The use of `?` in `cursor.execute()` is a best practice to safely pass variables to an SQL query, preventing common security vulnerabilities like SQL injection.
  * **Regular Expressions (`re`)**: A powerful tool for pattern matching in text. It is used here to reliably extract dates in a specific format from a user's query.
  * **Rule-Based Chatbot**: The chatbot's logic is based on a set of `if/else` rules that check for specific keywords or date patterns. This approach is simple to implement but lacks the flexibility of AI-driven chatbots.
  * **Use Cases**: This type of setup is practical for simple data logging, building prototypes for database applications, and creating basic conversational interfaces that retrieve information from structured data.
