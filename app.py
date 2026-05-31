import scrygent

def run_pipeline():
    print("Initializing scrygent pipeline...")
    
    # Call the default uv function
    message = scrygent.hello()
    print(f"Output: {message}")
    
    print("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()