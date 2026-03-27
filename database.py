import sqlite3

DB_FILE = "users.db"


def setup_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (name TEXT, age INTEGER, city TEXT)")
    c.execute("DELETE FROM users")
    c.executemany("INSERT INTO users VALUES (?, ?, ?)", [
        ("john", 25, "Mumbai"),
        ("alice", 30, "Delhi")
    ])
    conn.commit()
    conn.close()


def get_user(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name, age, city FROM users WHERE LOWER(name) = ?", (name.lower(),))
    user = c.fetchone()
    conn.close()

    if user:
        return f"{user[0].title()} is {user[1]} years old and lives in {user[2]}"
    return f"User '{name}' not found in database"


setup_database()
