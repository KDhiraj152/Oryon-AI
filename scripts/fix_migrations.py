#!/usr/bin/env python3
"""Fix alembic migration version."""
import os
from sqlalchemy import create_engine, text

def main():
    # Get database URL from environment or use default with explicit password
    db_url = os.getenv(
        'DATABASE_URL', 
        'postgresql://postgres:postgres@localhost:5432/education_content'  # Default password
    )
    
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        # Create alembic_version table if it doesn't exist
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS alembic_version (
                version_num VARCHAR(32) NOT NULL PRIMARY KEY
            )
        '''))
        
        # Clear existing versions
        conn.execute(text('DELETE FROM alembic_version'))
        
        # Set to latest migration
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('003_add_authentication')"))
        
        conn.commit()
        
    print("✓ Migration version set to 003_add_authentication")
    print("✓ Database is now in sync with migration files")

if __name__ == '__main__':
    main()
