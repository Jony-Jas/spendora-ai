import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import psycopg2

from dotenv import load_dotenv
load_dotenv()


class DbSaver:
    def __init__(self):
        self.embeddings = NVIDIAEmbeddings(
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            api_key=os.getenv("NVIDIA_API_KEY"),
            truncate="NONE",
        )
        self.vector_db = Chroma(
            embedding_function=self.embeddings, persist_directory="./vector_db")

    def save(self, input):

        if not isinstance(input, dict):
            print(type(input))
            raise ValueError("Input must be a dictionary.")
        
        self.save_to_sql(input)

        self.save_to_vector_db(input)

        return "Successfully saved the documents to database."

    def save_to_sql(self, input):
        db_config = {
            'dbname': 'postgres',  # Replace with your database name
            'user': 'postgres',         # Replace with your username
            'password': 'admin',     # Replace with your password
            'host': 'localhost',             # Replace with your host (if not local)
            'port': '5432'                   # Replace with your port (if not default)
        }
        conn = psycopg2.connect(**db_config)

        with conn.cursor() as cur:
            
            match input["action"]:
                case "CREATE":
                    cur.execute(
                        """
                        INSERT INTO expenses (expense_id, user_id, amount, category, date, description)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (input["expenseId"], input["userId"], input["amount"], input["category"], input["date"], input["description"])
                    )
                    conn.commit()
                    return
                case "UPDATE":
                    cur.execute(
                        """
                        UPDATE expenses
                        SET amount = %s, category = %s, date = %s, description = %s
                        WHERE expense_id = %s
                        """,
                        (input["amount"], input["category"], input["date"],  input["description"], input["expenseId"])
                    )
                    conn.commit()
                    return
                case "DELETE":
                    cur.execute(
                        """
                        DELETE FROM expenses
                        WHERE expense_id = %s
                        """,
                        (input["expenseId"],)
                    )
                    conn.commit()
                    return

        conn.close()
        print(f"Successfully {input['action']} the document to SQL database.")

    def save_to_vector_db(self, input):
        doc = Document(
            page_content=input["description"],
            metadata={
                "expenseId": input["expenseId"],
                "userId": input["userId"],
                "action": input["action"],
                "amount": input["amount"],
                "category": input["category"],
                "date": input["date"]
            }
        )

        match input["action"]:
            case "CREATE":
                print(self.vector_db.add_documents(
                    documents=[doc],
                    ids=[input["expenseId"]]
                ))
                return
            case "UPDATE":
                print(self.vector_db.update_documents(
                    documents=[doc],
                    ids=[input["expenseId"]]
                ))
                return
            case "DELETE":
                print(self.vector_db.delete(
                    ids=[input["expenseId"]]
                ))

        print("Successfully saved the document to vector database.")
