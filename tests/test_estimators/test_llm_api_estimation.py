"""
Tests to make sure the estimates for simulation actually match the actual cost.
"""

import pytest
from openai import OpenAI
from instructor import patch
from pydantic import BaseModel
from costly.estimators.llm_api_estimation import LLM_API_Estimation

def test_estimate_contains_exact():
    ...

