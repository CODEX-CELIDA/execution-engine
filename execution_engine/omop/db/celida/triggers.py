trigger_interval_overlap_check_function_sql = """
CREATE OR REPLACE FUNCTION check_overlapping_intervals()
RETURNS TRIGGER AS $$
DECLARE
    conflicting_row RECORD;
BEGIN
    SELECT INTO conflicting_row *
    FROM {schema}.{table}
    WHERE NEW.person_id = person_id
    AND NEW.run_id = run_id
    AND NEW.pi_pair_id IS NOT DISTINCT FROM pi_pair_id
    AND NEW.criterion_id IS NOT DISTINCT FROM criterion_id
    AND NEW.cohort_category IS NOT DISTINCT FROM cohort_category
    AND (NEW.interval_start, NEW.interval_end) OVERLAPS (interval_start, interval_end)
    LIMIT 1;

    IF FOUND THEN
        RAISE EXCEPTION 'Overlapping intervals detected.'
        USING DETAIL = format(
            'Existing row - person_id: %s, run_id: %s, pi_pair_id: %s, criterion_id: %s, cohort_category: %s, interval_start: %s, interval_end: %s;\n' ||
            '              New row - person_id: %s, run_id: %s, pi_pair_id: %s, criterion_id: %s, cohort_category: %s, interval_start: %s, interval_end: %s',
            conflicting_row.person_id, conflicting_row.run_id, conflicting_row.pi_pair_id, conflicting_row.criterion_id, conflicting_row.cohort_category, conflicting_row.interval_start, conflicting_row.interval_end,
            NEW.person_id, NEW.run_id, NEW.pi_pair_id, NEW.criterion_id, NEW.cohort_category, NEW.interval_start, NEW.interval_end
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

"""
create_trigger_interval_overlap_check_sql = """
CREATE TRIGGER trigger_result_interval_before_insert
BEFORE INSERT ON {schema}.{table}
FOR EACH ROW EXECUTE FUNCTION check_overlapping_intervals();
"""
