from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Columns added after the first release. create_all() only creates missing
# tables, so older databases get these columns added here.
_ADDED_COLUMNS = {
    "applications": {"tenure_months": "INTEGER NOT NULL DEFAULT 12", "recourse_preferences": "JSON"},
}


def upgrade_schema(bind):
    insp = inspect(bind)
    with bind.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if not insp.has_table(table):
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def init_db(bind=engine):
    """Initialize database tables"""
    Base.metadata.create_all(bind=bind)
    upgrade_schema(bind)
