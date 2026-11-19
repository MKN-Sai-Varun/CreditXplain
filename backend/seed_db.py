"""
Seed database with demo users for testing
Run: python3 -m backend.seed_db
"""

from backend.database import SessionLocal, init_db
from backend.models import User, UserRole
from backend.auth import get_password_hash

def seed_demo_users():
    """Create demo users (applicant and officer)"""
    init_db()
    db = SessionLocal()
    
    try:
        # Check if users already exist
        existing_applicant = db.query(User).filter(User.email == "applicant@creditxplain.com").first()
        existing_officer = db.query(User).filter(User.email == "officer@creditxplain.com").first()
        
        if existing_applicant and existing_officer:
            print("✓ Demo users already exist")
            return
        
        # Create demo applicant
        if not existing_applicant:
            applicant = User(
                email="applicant@creditxplain.com",
                password_hash=get_password_hash("password123"),
                role=UserRole.APPLICANT,
                full_name="Demo Applicant"
            )
            db.add(applicant)
            print("✓ Created demo applicant: applicant@creditxplain.com / password123")
        
        # Create demo officer
        if not existing_officer:
            officer = User(
                email="officer@creditxplain.com",
                password_hash=get_password_hash("password123"),
                role=UserRole.OFFICER,
                full_name="Demo Officer"
            )
            db.add(officer)
            print("✓ Created demo officer: officer@creditxplain.com / password123")
        
        db.commit()
        print("\n✅ Database seeded successfully!")
        print("\nDemo Credentials:")
        print("  Applicant: applicant@creditxplain.com / password123")
        print("  Officer:   officer@creditxplain.com / password123")
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_demo_users()
