"""Pydantic schemas. Fields that may be absent in the text are Optional."""
import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    vendor_name: Optional[str] = None
    total_amount: Optional[float] = Field(None, ge=0)
    currency: Optional[str] = None

    @field_validator("date")
    @classmethod
    def check_date(cls, v):
        if v is None:
            return v
        try:
            return datetime.strptime(v, "%Y-%m-%d").date().isoformat()
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")

    @field_validator("total_amount", mode="before")
    @classmethod
    def clean_amount(cls, v):
        # Accept "1,250.00" or "$1,250.00" as well as plain numbers
        if isinstance(v, str):
            v = re.sub(r"[^\d.\-]", "", v)
            return float(v) if v else None
        return v


class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


class Experience(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None


class Resume(BaseModel):
    candidate_name: Optional[str] = None
    email: Optional[EmailStr] = None
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)


SCHEMAS = {"invoice": Invoice, "resume": Resume}