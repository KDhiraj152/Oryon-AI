"""Initialize database with all tables from SQLAlchemy models"""
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

import database
import models  # Import all models to register them with Base


def init_database():
    """Create all tables in the database"""
    print("🔨 Creating all database tables...")
    print(f"📍 Database URL: {database.engine.url}")
    database.Base.metadata.create_all(bind=database.engine)
    print("✅ Database initialized successfully!")
    
    # Show created tables
    from sqlalchemy import inspect
    inspector = inspect(database.engine)
    tables = inspector.get_table_names()
    print(f"📊 Created {len(tables)} tables:")
    for table in sorted(tables):
        print(f"   - {table}")

if __name__ == "__main__":
    init_database()
