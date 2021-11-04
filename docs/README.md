# Quick Docs Intro

If you have installed floodlight from source and set up your dev environment via poetry, just run the sphinx build from within this directory to build the docs locally:

```
cd docs
poetry run sphinx-build -b html source build/html
```

You can now check out the docs at `docs/build/html/index.html`
