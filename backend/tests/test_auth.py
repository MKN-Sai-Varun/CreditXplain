import pytest
from backend.auth import get_password_hash, verify_password, create_access_token
from jose import jwt
from backend.config import settings

def test_password_hashing():
    """Test password hashing and verification"""
    password = "testpassword123"
    hashed = get_password_hash(password)
    
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False

def test_create_access_token():
    """Test JWT token creation"""
    data = {"sub": "test@example.com"}
    token = create_access_token(data)
    
    assert token is not None
    assert len(token) > 0
    
    # Decode and verify
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    assert payload["sub"] == "test@example.com"
    assert "exp" in payload

def test_password_hash_uniqueness():
    """Test that same password generates different hashes"""
    password = "testpassword123"
    hash1 = get_password_hash(password)
    hash2 = get_password_hash(password)
    
    # Hashes should be different (bcrypt uses salt)
    assert hash1 != hash2
    # But both should verify
    assert verify_password(password, hash1) is True
    assert verify_password(password, hash2) is True
