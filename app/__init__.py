"""Dashboard package (Step 7 of the build plan).

Not yet implemented. Planned: a Streamlit app for interactive inference --
upload/paste a sequence, view the predicted taxonomy with the hierarchical
confidence vector, and see a novelty flag when applicable. Will be served
by a separate, lightweight ``Dockerfile.inference`` image that pulls model
weights from external storage (HuggingFace Hub/S3/GitHub Release) at
runtime rather than storing them in git.
"""
