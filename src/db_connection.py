import psycopg2


def main():
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="Intelligent Ticket Assignment & Workload Management",
            user="postgres",
            password="2014"
        )

        cur = conn.cursor()
        print("Database connected successfully")

        # Count total rows
        cur.execute("SELECT COUNT(*) FROM autotask_raw;")
        count = cur.fetchone()
        print("Total rows in autotask_raw:", count[0])

        # Fetch sample rows
        cur.execute("""
            SELECT * FROM autotask_raw
            LIMIT 5;
        """)
        rows = cur.fetchall()

        print("\nSample rows:")
        for row in rows:
            print(row)

        cur.close()
        conn.close()

        print("\nDatabase connection closed")

    except Exception as error:
        print("Connection failed:", error)


if __name__ == "__main__":
    main()
    