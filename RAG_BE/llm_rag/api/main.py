from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import argparse
import yaml
from llm_rag.core.rag_main import Rag_config, rag
import os
import json

rag_instance = None


def load_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            rag_config = Rag_config(**data)
            return rag_config
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error loading YAML file '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def parse_command_line_script():
    parser = argparse.ArgumentParser(description="Load YAML file using argparse")
    parser.add_argument("run", help="Command to run", choices=["run"])
    parser.add_argument(
        "-c", "--config", help="Path to YAML config file", required=True
    )
    args = parser.parse_args()

    if args.run == "run":
        config_path = args.config
        config_data = load_yaml(config_path)
        if config_data:
            print("Config data loaded successfully:")
            print(config_data)
            return config_data
        else:
            raise ValueError("Config Not Found")
    else:
        raise ValueError("run not put in command line")


def get_json() -> None:
    if os.path.exists("data.json"):
        # If file.json exists, load its contents
        with open("data.json", "r") as f:
            data = json.load(f)
    else:
        # If file.json doesn't exist, initialize an empty list
        data = dict()
    return data


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance
    rag_config = parse_command_line_script()
    rag_instance = rag(rag_config)
    yield


app = FastAPI(lifespan=lifespan)

# Allow all origins for CORS. You can restrict it to specific origins if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status/")
async def get_status()->dict[str,str]:
    return {"status": "Server is running"}


@app.post("/api/upload-file/")
async def upload_file(file: UploadFile = File(...))->dict[str,str]:
    if rag_instance is None:
        raise ValueError("Instance is None")
    contents = await file.read()
    rag_instance.insert_file(contents, file.filename)
    return {"filename": f"{file.filename} inserted successfully"}


@app.get("/api/list-files")
async def list_files()->list[dict[str,str]]:
    data = get_json()
    new_list = []
    for filename, info in data.items():
        new_list.append({
            "filename": filename,
            "last_modified_time": info["last_modified_time"]
        })

    return new_list


@app.delete("/api/delete-file/")
async def delete_file(filename: str)->dict[str,str]:
    if rag_instance is None:
        raise ValueError("Instance is None")
    msg = "Deleted succesfully"
    try:
        rag_instance.delete_file(filename)
    except Exception as e:
        msg = f"delete did not occur, due to {e}"
    return {"msg": f"{filename} {msg}"}


@app.get("/api/chat")
async def chat(query: str):
    if rag_instance is None:
        raise ValueError("Instance is None")
    response = rag_instance.query(query)
    return {"response": response}


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
