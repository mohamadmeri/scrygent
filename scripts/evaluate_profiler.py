import json
import logging
from pathlib import Path

from scrygent.tools.io import load_csv
from scrygent.tools.profiler import profile_dataframe

logging.basicConfig(level=logging.INFO)

def run_planner_emulation():
    # Target a real DABench table from your existing tree
    csv_path = Path("data/InfiAgent/data/da-dev-tables/titanic.csv")
    
    if not csv_path.exists():
        logging.error("Could not find %s. Verify the path.", csv_path)
        return

    # A realistic DABench query that tests the regex and semantic limits
    user_query = "What is the survival rate of female passengers who paid a fare greater than $50?"
    
    # 1. Load data
    df = load_csv(csv_path)
    
    # 2. Profile
    profile_output = profile_dataframe(df, user_query)
    
    # 3. Print the payload exactly as the Planner Node will see it
    print("\n" + "="*80)
    print(f"FILE: {csv_path.name}")
    print(f"USER QUERY: {user_query}")
    print("="*80)
    print("PROFILER OUTPUT (The LLM's Context Window):")
    print("="*80)
    print(json.dumps(profile_output, indent=2))
    print("="*80)

if __name__ == "__main__":
    run_planner_emulation()
