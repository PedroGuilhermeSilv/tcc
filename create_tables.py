# create_tables.py
from src.database import Base, engine

# Import all your models here
from src.converter.model.entity import Converter, Pet

if __name__ == "__main__":
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")
