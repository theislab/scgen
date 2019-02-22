# scGen Documentation

The source for scGen documentation is in this directory under `sources/`. 
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- install MkDocs: `pip install mkdocs`
- `cd` to the `docs/` folder and run:
- `python generate_docs.py`
- `mkdocs serve`    # Starts a local web-server:  [localhost:8000](localhost:8000)
- `mkdocs build`    # Builds a static site in "site" directory
