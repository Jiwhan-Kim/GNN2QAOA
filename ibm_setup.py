import os
from dotenv import load_dotenv

from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()

token = os.getenv("IBM_TOKEN")
instance = os.getenv("IBM_INSTANCE")

if token is None:
    print("Error: IBM_TOKEN is not found. Please read README.md")

QiskitRuntimeService.save_account(
    token=token,
    instance=instance
)
