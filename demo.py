import ollama
import pymongo
import time

desiredModel = 'llama3.2'
MDB_URI = ""
# Demo setup
def parse_json_to_text(data):
    print(f"Parsing {len(data)} documents to text...")
    texts = []
    for doc in data:
        title = doc.get('title', 'N/A')
        text = (
            f"Title: {title}\n"
            "-----\n"
        )
        texts.append(text)
    return "\n".join(texts)

def prepare_prompt(context, batch_size):
    prompt = (
        "Given the [context]\n\n"
        f"[context]\n{context}\n"
        "\n[/context]\n\n"
        "RESPOND WITH A `LIST OF THE MOVIE TITLES IN THE [context];` "
        "LIST OF TITLES ONLY!, SEPARATED BY `\n` and double quotes! ESCAPE QUOTES WHEN NEEDED! "
        "YOU MUST ONLY USE [context] TO FORMULATE YOUR RESPONSE! MAKE SURE YOU RESPOND IN THE CORRECT FORMAT OR I WON'T BE ABLE TO UNDERSTAND YOUR RESPONSE!"
    )
    
    system_message = """
You will receive some [context], and your task is to respond with a list of movie titles in the [context] separated by `\n` and double quotes.
It is very important that you respond in the correct format, and never wrap your response in ``` or `.
You must only use [context] to formulate your response. Make sure you respond in the correct format or I won't be able to understand your response.         
IMPORTANT! ALWAYS SEPARATE USING \n!    
[response_format]
"movie title goes here"\n
"movie title goes here"\n
"movie title goes here"\n
"movie title goes here"\n
[/response_format]   

- Must respond in the correct format, and never wrap your response in ``` or `.
- You must only use [context] to formulate your response.
- Make sure you respond in the correct format or I won't be able to understand your response.
- ALWAYS SEPARATE USING \n! A SINGLE NEWLINE! ONE TITLE PER LINE! IMPORTANT!
THINK STEP BY STEP.
    """
    
    user_instructions = f"""
[response_format]
"movie title goes here"\n
"movie title goes here"\n
"movie title goes here"\n
"movie title goes here"\n
[/response_format]

IMPORTANT! ONLY RESPOND IN THIS FORMAT! ONE TITLE PER LINE!
----------------------------------------------------
THINK CRITICALLY AND STEP BY STEP. EXPECTED LIST SIZE: {batch_size}
IMPORTANT! ALWAYS SEPARATE USING newline \n! NEVER SEPARATE USING SPACES, COMMAS, OR ANYTHING ELSE.
    """
    
    return system_message, prompt, user_instructions

def send_request_to_model(context, batch_size, desiredModel, max_retries=10, backoff_factor=0):
    system_message, prompt, user_instructions = prepare_prompt(context, batch_size)
    
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(
                model=desiredModel,
                messages=[
                    {
                        'role': 'system',
                        'content': system_message
                    },
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                    {
                        'role': 'user',
                        'content': user_instructions,
                    },
                ]
            )
            
            if response.get('message') and response['message'].get('content'):
                print(f"\nResponse received on attempt {attempt}:")
                print(response['message']['content'])
                
                # Parse the response
                csv_output = response['message']['content']
                csv_lines = [line.strip() for line in csv_output.split('\n') if line.strip() and line.strip() not in {'```', '.', '`'}]
                
                # Remove surrounding quotes if present
                csv_lines = [line.strip('"') for line in csv_lines]
                
                if len(csv_lines) == batch_size:
                    print(f"Batch processed successfully with {len(csv_lines)} titles.")
                    return csv_lines
                else:
                    print(f"Attempt {attempt}: Mismatch in expected batch size. Expected {batch_size}, got {len(csv_lines)}.")
            else:
                print(f"Attempt {attempt}: No response content received.")
                
        except Exception as e:
            print(f"Attempt {attempt}: An error occurred while communicating with the model: {e}")
        
        if attempt < max_retries:
            sleep_time = backoff_factor ** attempt
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            print(f"Attempt {attempt}: Max retries reached. Moving to next batch.")
    
    return []  # Return empty list if all retries fail

def main():
    try:
        # Load text from MongoDB
        print("Connecting to MongoDB...")
        client = pymongo.MongoClient(MDB_URI)
        db = client["sample_mflix"]
        collection = db["movies"]
        print("Connected to MongoDB. Fetching data...")
        
        formatted_text = list(collection.aggregate([
            {"$match": {}},
            {"$project": {"title": 1}},
            {"$limit": 1000}
        ]))
        
        print(f"Fetched {len(formatted_text)} documents from MongoDB.")
        
        tmp_batch = []
        total_docs = 0
        final_result = []
        batch_round = 0
        batch_size = 20
        failed_batches = []
        
        for i, doc in enumerate(formatted_text, 1):
            tmp_batch.append(doc)
            
            if i % batch_size == 0:
                context = parse_json_to_text(tmp_batch)
                csv_lines = send_request_to_model(context, batch_size, desiredModel)
                
                if csv_lines:
                    final_result.extend(csv_lines)
                    total_docs += len(csv_lines)
                    batch_round += 1
                    print(f"Batch {batch_round}: Total documents processed: {total_docs}")
                else:
                    # Store the failed batch for retrying later
                    failed_batches.append(list(tmp_batch))
                    print(f"Batch {batch_round + 1}: Failed to process this batch. Will retry later.")
                
                tmp_batch = []
        
        # Process the last batch if it's not empty and not a full batch
        if tmp_batch:
            context = parse_json_to_text(tmp_batch)
            last_batch_size = len(tmp_batch)
            csv_lines = send_request_to_model(context, last_batch_size, desiredModel)
            
            if csv_lines:
                final_result.extend(csv_lines)
                total_docs += len(csv_lines)
                batch_round += 1
                print(f"Final Batch {batch_round}: Total documents processed: {total_docs}")
            else:
                failed_batches.append(list(tmp_batch))
                print(f"Final Batch {batch_round + 1}: Failed to process this batch. Will retry later.")
        
        # Retry failed batches
        if failed_batches:
            print(f"\nRetrying {len(failed_batches)} failed batch(es)...")
            remaining_failed_batches = []
            for idx, batch in enumerate(failed_batches, 1):
                print(f"\nRetrying Batch {idx} of {len(failed_batches)}:")
                context = parse_json_to_text(batch)
                current_batch_size = len(batch)
                csv_lines = send_request_to_model(context, current_batch_size, desiredModel)
                
                if csv_lines:
                    final_result.extend(csv_lines)
                    total_docs += len(csv_lines)
                    batch_round += 1
                    print(f"Retry Batch {idx}: Total documents processed: {total_docs}")
                else:
                    remaining_failed_batches.append(batch)
                    print(f"Retry Batch {idx}: Failed again. Will skip this batch.")
            
            # If there are still failed batches, log them
            if remaining_failed_batches:
                print(f"\n{len(remaining_failed_batches)} batch(es) failed after retries. Please inspect manually.")
                for idx, batch in enumerate(remaining_failed_batches, 1):
                    print(f"\nFailed Batch {idx}:")
                    for doc in batch:
                        print(doc.get('title', 'N/A'))
            else:
                print("\nAll batches processed successfully after retries.")
        
        # Final validation
        if len(final_result) == 1000:
            print("\nAll 1000 movie titles have been successfully collected.")
        else:
            print(f"\nTotal titles collected: {len(final_result)} out of 1000.")
            missing = 1000 - len(final_result)
            print(f"Missing {missing} titles.")
        
        # Optional: Save the results to a file
        # with open("movie_titles.txt", "w", encoding="utf-8") as f:
        #     for title in final_result:
        #         f.write(f"{title}\n")
        
        # Print the final result
        print("\nHere is the complete list of movie titles:")
        for idx, title in enumerate(final_result, 1):
            print(f"{idx}: {title}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
