"""Shared, orchestrator-agnostic code for the covertype data pipeline.

Kept out of the DAG files on purpose: importing one DAG file from another
makes the scheduler register both DAGs under the importing file.
"""
