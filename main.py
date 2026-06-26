import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database import init_relational_database
from src.vector_db import init_vector_database
from src.telegram_bot import main as start_bot

if __name__ == "__main__":
    print("============")
    print("FINAGENT PLATFORM")
    print("============")
    
    init_relational_database()
    
    init_vector_database()
    
    start_bot()