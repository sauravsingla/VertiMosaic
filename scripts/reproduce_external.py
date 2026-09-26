"""External-data preparation entry point.

Retrieval remains explicit because external source terms and network access are environment-dependent.
"""

from vertimosaic.datasets import list_specs

if __name__ == "__main__":
    print("Registered external sources:")
    for spec in list_specs():
        print(spec)
    print("Use `vertimosaic datasets download <key>` to retrieve a source explicitly.")
