while IFS= read -r sample_id; do
    echo "running data processing for sample: $sample_id"
    python scanpy_processing.py --sample_id $sample_id
done < ../sample_ids