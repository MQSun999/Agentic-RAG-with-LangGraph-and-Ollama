from enum import Enum 


from typing import Optional, List
from pydantic import BaseModel, Field

# Enum enables you to select only one value out of many
class DocType(Enum):
    TEN_K = "10-k"
    TEN_Q = "10-q"
    EIGHT_K = "8-k"
    OTHER = "other"


class FiscalQuarter(Enum):
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"



# ============================================================================
# PYDANTIC MODELS
# ============================================================================
# 
class ChunkMetadata(BaseModel):
    
    company_name: Optional[str] = Field(default=None, description="Company name (lowercase, eg. 'amazon', 'apple', 'google',...)")
    doc_type: Optional[DocType] =  Field(default=None, description="Document type (10-k, 10-q, 8-k, etc.)")
    fiscal_year: Optional[int] = Field(default=None, ge=1950, le=2050, description="Fiscal year of the document")
    fiscal_quarter: Optional[FiscalQuarter] = Field(default=None, description="Fiscal quarter (q1-q4) if applicable")

    model_config = {"use_enum_values": True}


class RankingKeywords(BaseModel):
    """Must return an object with a keywords field, NOT a plain list."""
    keywords: List[str] = Field(..., description="Generate Exactly 5 financial keywords related to user query", min_length=5, max_length=5)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the query, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded with the facts for the query, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses query."""

    binary_score: str = Field(description="Answer addresses the query, 'yes' or 'no'")


class SearchQueries(BaseModel):
    """Search queries for retrieving missing information."""

    search_queries: list[str] = Field(
        description="1-3 search queries to retrieve the missing information."
    )
